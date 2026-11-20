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
using Distributions

const MEM = MacroEconometricModels
if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "Non-Gaussian Internals" begin

    # =========================================================================
    # ICA Internals (src/nongaussian/ica.jl)
    # =========================================================================

    @testset "Whitening" begin
        rng = Xoshiro(7001)
        # Identity-covariance input should yield near-identity output covariance
        U = randn(rng, 200, 3)
        Z, W_white, dewhiten = MEM._whiten(U)
        @test size(Z) == (200, 3)
        @test size(W_white) == (3, 3)
        @test size(dewhiten) == (3, 3)
        cov_Z = Z' * Z / size(Z, 1)
        @test norm(cov_Z - I) < 0.3  # approximately identity

        # Near-zero eigenvalue: rank-deficient input
        U_rank = hcat(randn(rng, 100, 2), zeros(100))
        Z2, W2, dw2 = MEM._whiten(U_rank)
        @test size(Z2, 1) == 100
        @test size(Z2, 2) <= 3  # may drop near-zero eigenvalue component
    end

    @testset "Givens rotation roundtrip" begin
        rng = Xoshiro(7002)
        n = 3
        n_angles = n * (n - 1) ÷ 2
        angles_orig = randn(rng, n_angles)
        Q = MEM._givens_to_orthogonal(angles_orig, n)
        @test size(Q) == (n, n)
        # Q should be orthogonal
        @test norm(Q' * Q - I) < 1e-10
        @test norm(Q * Q' - I) < 1e-10

        # Roundtrip: extract angles and reconstruct (exact for Q ∈ SO(n))
        angles_extracted = MEM._orthogonal_to_givens(Q, n)
        Q_rebuilt = MEM._givens_to_orthogonal(angles_extracted, n)
        @test norm(Q_rebuilt' * Q_rebuilt - I) < 1e-12
        @test norm(Q - Q_rebuilt) < 1e-12

        for ndim in (2, 3, 4)
            Qr = MEM.generate_Q(ndim, Float64; rng=rng)
            if det(Qr) < 0
                Qr[:, ndim] .*= -1.0
            end
            ang_r = MEM._orthogonal_to_givens(Qr, ndim)
            Q_rec = MEM._givens_to_orthogonal(ang_r, ndim)
            @test norm(Qr - Q_rec) < 1e-12
        end

        # n=2 case
        angles2 = [0.5]
        Q2 = MEM._givens_to_orthogonal(angles2, 2)
        @test size(Q2) == (2, 2)
        @test norm(Q2' * Q2 - I) < 1e-10
    end

    @testset "Contrast functions" begin
        x = 1.5

        # logcosh: G(u) = log(cosh(u)), g(u) = tanh(u), g'(u) = 1 - tanh^2(u)
        G, g, gp = MEM._contrast_logcosh(x)
        @test G ≈ log(cosh(x))
        @test g ≈ tanh(x)
        @test gp ≈ 1 - tanh(x)^2

        # exp: G(u) = -exp(-u^2/2), g(u) = u*exp(-u^2/2)
        G_e, g_e, gp_e = MEM._contrast_exp(x)
        @test G_e ≈ -exp(-x^2 / 2)
        @test g_e ≈ x * exp(-x^2 / 2)
        @test gp_e ≈ (1 - x^2) * exp(-x^2 / 2)

        # kurtosis: G(u) = u^4/4, g(u) = u^3, g'(u) = 3u^2
        G_k, g_k, gp_k = MEM._contrast_kurtosis(x)
        @test G_k ≈ x^4 / 4
        @test g_k ≈ x^3
        @test gp_k ≈ 3 * x^2

        # _get_contrast dispatch
        @test MEM._get_contrast(:logcosh) === MEM._contrast_logcosh
        @test MEM._get_contrast(:exp) === MEM._contrast_exp
        @test MEM._get_contrast(:kurtosis) === MEM._contrast_kurtosis
        @test_throws ArgumentError MEM._get_contrast(:unknown)
    end

    @testset "FastICA deflation" begin
        rng = Xoshiro(7003)
        Z = randn(rng, 100, 3)
        # Whiten it properly
        Z_w, _, _ = MEM._whiten(Z)
        n = size(Z_w, 2)
        W, iters = MEM._fastica_deflation(Z_w, n; contrast=:logcosh, max_iter=50)
        @test size(W) == (n, n)
        @test iters > 0
        # W rows should be approximately unit norm
        for i in 1:n
            @test abs(norm(W[i, :]) - 1.0) < 0.1
        end
    end

    @testset "FastICA symmetric" begin
        rng = Xoshiro(7004)
        Z = randn(rng, 100, 3)
        Z_w, _, _ = MEM._whiten(Z)
        n = size(Z_w, 2)
        W, iters = MEM._fastica_symmetric(Z_w, n; contrast=:exp, max_iter=50)
        @test size(W) == (n, n)
        @test iters > 0
        @test iters <= 50
    end

    @testset "ICA to SVAR conversion" begin
        rng = Xoshiro(7005)
        Y = randn(rng, 200, 3)
        model = estimate_var(Y, 2)
        n = 3
        W_ica = Matrix{Float64}(I, n, n)  # identity rotation
        B0, Q, shocks = MEM._ica_to_svar(W_ica, model)
        @test size(B0) == (n, n)
        @test size(Q) == (n, n)
        @test norm(Q' * Q - I) < 1e-6
        # Diagonal of B0 should be positive (sign normalization)
        @test all(diag(B0) .>= 0)
    end

    @testset "JADE cumulant matrices" begin
        rng = Xoshiro(7006)
        Z = randn(rng, 100, 3)
        Z_w, _, _ = MEM._whiten(Z)
        mats = MEM._jade_cumulant_matrices(Z_w)
        n = size(Z_w, 2)
        n_mats = n * (n + 1) ÷ 2
        @test length(mats) == n_mats
        @test all(size(m) == (n, n) for m in mats)
    end

    @testset "Joint diagonalization" begin
        # Diagonal matrices should converge quickly
        n = 3
        D1 = Diagonal([1.0, 2.0, 3.0])
        D2 = Diagonal([4.0, 5.0, 6.0])
        mats = [Matrix(D1), Matrix(D2)]
        V, iters = MEM._joint_diagonalization(mats; max_iter=50, tol=1e-8)
        @test size(V) == (n, n)
        @test iters <= 50
        # V should be close to identity (or permutation) for diagonal input
        @test norm(V' * V - I) < 0.1
    end

    @testset "SOBI autocovariance" begin
        rng = Xoshiro(7008)
        Z = randn(rng, 100, 3)
        R = MEM._sobi_autocovariance(Z, 1)
        @test size(R) == (3, 3)
        R0 = MEM._sobi_autocovariance(Z, 0)
        @test size(R0) == (3, 3)
    end

    @testset "Distance covariance" begin
        rng = Xoshiro(7009)
        # Independent variables: low distance covariance
        x = randn(rng, 50)
        y = randn(rng, 50)
        dcov_ind = MEM._distance_covariance(x, y)
        @test dcov_ind >= 0

        # Perfectly dependent: high distance covariance
        dcov_dep = MEM._distance_covariance(x, x .^ 2 .+ 0.01 * randn(rng, 50))
        @test dcov_dep >= 0
    end

    @testset "dCov objective" begin
        rng = Xoshiro(7010)
        Z = randn(rng, 50, 2)
        angles = zeros(1)  # 2x2 case: 1 angle
        obj = MEM._dcov_objective(angles, Z, 2)
        @test obj >= 0
        # T159: kept — nonnegativity above is the pin for a dependence measure.
        @test isfinite(obj)
    end

    @testset "HSIC statistic" begin
        rng = Xoshiro(7011)
        x = randn(rng, 30)
        y = randn(rng, 30)
        hsic_val = MEM._hsic_statistic(x, y; sigma=1.0)
        @test isfinite(hsic_val)
        @test hsic_val >= 0  # T159: tr(Kc·Lc) of psd kernels (observed 0.0055; mirrors the dcov pin)
    end

    @testset "HSIC objective" begin
        rng = Xoshiro(7012)
        Z = randn(rng, 30, 2)
        angles = zeros(1)
        obj = MEM._hsic_objective(angles, Z, 2; sigma=1.0)
        @test isfinite(obj)
        @test obj >= 0  # T159: sum of HSIC terms (observed 0.0052)
    end

    # =========================================================================
    # ML Internals (src/nongaussian/ml.jl)
    # =========================================================================

    @testset "Student-t log-pdf" begin
        # Should return finite values (own unit-variance parameterization — NOT Distributions.TDist)
        val = MEM._student_t_logpdf(0.0, 5.0)
        @test isfinite(val)
        @test val ≈ -0.7132067771717288  # T159: deterministic pure-function value
        # nu near boundary (2.01)
        val_boundary = MEM._student_t_logpdf(1.0, 2.01)
        @test isfinite(val_boundary)
        @test val_boundary ≈ -5.333258781178699  # T159: deterministic pure-function value
        # Large nu should approximate normal
        val_large = MEM._student_t_logpdf(0.0, 1000.0)
        normal_val = logpdf(Normal(), 0.0)
        @test abs(val_large - normal_val) < 0.1
    end

    @testset "Mixture normal log-pdf" begin
        # p_mix = 0.5, equal variances => standard normal
        val = MEM._mixture_normal_logpdf(0.0, 0.5, 1.0, 1.0)
        @test isfinite(val)
        @test val ≈ logpdf(Normal(), 0.0)  # T159: exact reduction (observed to fp)
        # p_mix = 0.8, sigma1=0.5, sigma2=1.5
        val_asym = MEM._mixture_normal_logpdf(0.0, 0.8, 0.5, 1.5)
        @test isfinite(val_asym)
        @test val_asym ≈ log(0.8 * pdf(Normal(0, 0.5), 0.0) + 0.2 * pdf(Normal(0, 1.5), 0.0))  # T159: closed form (observed to fp)
        # p_mix near 0 => mostly second component
        val_low = MEM._mixture_normal_logpdf(0.0, 0.01, 0.5, 1.0)
        @test isfinite(val_low)
        @test val_low ≈ logpdf(Normal(), 0.0) atol = 0.05  # T159: 1% contamination (observed dev 0.0099)

        # 6-arg asymmetric mixture normal log-pdf
        @test MEM._mixture_normal_logpdf(0.0, 0.5, 0.0, 1.0, 0.0, 1.0) ≈ logpdf(Normal(), 0.0)
        val6 = MEM._mixture_normal_logpdf(1.0, 0.3, -0.5, 0.8, 0.7, 1.2)
        @test val6 ≈ log(0.3 * pdf(Normal(-0.5, 0.8), 1.0) + 0.7 * pdf(Normal(0.7, 1.2), 1.0))
        # Logsumexp tail stability: non-zero finite gradient at x = 20.0
        dx_tail = MEM.ForwardDiff.derivative(
            x -> MEM._mixture_normal_logpdf(x, 0.3, -0.5, 0.8, 0.7, 1.2), 20.0)
        @test isfinite(dx_tail) && dx_tail != 0.0
    end

    @testset "Skew-normal log-pdf" begin
        # alpha = 0 reduces to normal
        val_normal = MEM._skew_normal_logpdf(0.0, 0.0)
        expected = log(2.0) + logpdf(Normal(), 0.0) + logcdf(Normal(), 0.0)
        @test val_normal ≈ expected
        # Non-zero alpha
        val_skew = MEM._skew_normal_logpdf(1.0, 2.0)
        @test isfinite(val_skew)
        @test val_skew ≈ log(2.0) + logpdf(Normal(), 1.0) + logcdf(Normal(), 2.0)  # T159: closed form (observed to fp)
    end

    @testset "Pearson IV log-pdf" begin
        val = MEM._pearson_iv_logpdf(0.0, 0.0, 5.0)
        @test isfinite(val)
        @test val ≈ -0.8210027579468608  # T159: deterministic pure-function value
        val2 = MEM._pearson_iv_logpdf(1.0, 0.5, 10.0)
        @test isfinite(val2)
        @test val2 ≈ -1.439755024852729  # T159: deterministic pure-function value

        # Dual-safe _re_loggamma matches SpecialFunctions.loggamma(complex(...))
        for m in (2.05, 5.0, 15.9999, 16.0001, 40.0), y in (-25.0, 0.0, 25.0)
            ref = MEM._re_loggamma(m, y)
            d_val = MEM.ForwardDiff.value(
                MEM._re_loggamma(MEM.ForwardDiff.Dual(m, 1.0), MEM.ForwardDiff.Dual(y, 0.0)))
            @test abs(d_val - ref) < 1e-13
        end
        # Mixed-type promotion for _re_loggamma
        ref52 = MEM._re_loggamma(5.0, 2.0)
        @test abs(MEM.ForwardDiff.value(MEM._re_loggamma(MEM.ForwardDiff.Dual(5.0, 1.0), 2.0)) - ref52) < 1e-13
        @test abs(MEM.ForwardDiff.value(MEM._re_loggamma(5.0, MEM.ForwardDiff.Dual(2.0, 1.0))) - ref52) < 1e-13
        @test abs(MEM._re_loggamma(5.0f0, 2.0) - ref52) < 1e-6
        @test abs(MEM._re_loggamma(5, 2.0) - ref52) < 1e-13
        @test abs(MEM._re_loggamma(5, 2) - ref52) < 1e-13
        # ForwardDiff gradient of Pearson IV log-pdf vs finite differences
        p_piv = [0.7, 0.4, 5.0]
        f_piv = v -> MEM._pearson_iv_logpdf(v[1], v[2], v[3])
        g_piv = MEM.ForwardDiff.gradient(f_piv, p_piv)
        eps_fd = 1e-6
        g_piv_fd = [(f_piv(p_piv .+ eps_fd .* Matrix(1.0I, 3, 3)[:, i]) -
                     f_piv(p_piv .- eps_fd .* Matrix(1.0I, 3, 3)[:, i])) / (2 * eps_fd) for i in 1:3]
        @test maximum(abs.(g_piv - g_piv_fd)) < 1e-6
    end

    @testset "Gaussian log-likelihood" begin
        rng = Xoshiro(7020)
        Y = randn(rng, 100, 2)
        model = estimate_var(Y, 1)
        ll = MEM._gaussian_loglik(model)
        # T159: kept — the < 0 pin below is the real guard.
        @test isfinite(ll)
        @test ll < 0  # log-likelihood should be negative for reasonable data
    end

    @testset "_n_dist_params" begin
        @test MEM._n_dist_params(:student_t) == 1
        @test MEM._n_dist_params(:mixture_normal) == 2
        @test MEM._n_dist_params(:mixture_normal_asym) == 3
        @test MEM._n_dist_params(:pml) == 2
        @test MEM._n_dist_params(:skew_normal) == 1
        @test_throws ArgumentError MEM._n_dist_params(:unknown)
    end

    @testset "Pack/unpack nongaussian params roundtrip" begin
        angles = [0.1, 0.2, 0.3]
        dp = [1.0, 2.0]
        packed = MEM._pack_nongaussian_params(angles, dp)
        @test packed == [0.1, 0.2, 0.3, 1.0, 2.0]
        angles2, dp2 = MEM._unpack_nongaussian_params(packed, 3, :student_t)
        @test angles2 == angles
        @test dp2 == dp
    end

    @testset "Numerical Hessian" begin
        # f(x) = x'x => Hessian = 2I
        f(x) = dot(x, x)
        x0 = [1.0, 2.0, 3.0]
        H = MEM._numerical_hessian(f, x0)
        @test size(H) == (3, 3)
        @test norm(H - 2 * I) < 0.01  # Should be close to 2I

        # f(x) = x[1]^2 + 2*x[2]^2 => H = diag(2, 4)
        g(x) = x[1]^2 + 2 * x[2]^2
        H2 = MEM._numerical_hessian(g, [0.0, 0.0])
        @test abs(H2[1, 1] - 2.0) < 0.01
        @test abs(H2[2, 2] - 4.0) < 0.01
        @test abs(H2[1, 2]) < 0.01
    end

    @testset "Non-Gaussian loglik branches" begin
        rng = Xoshiro(7025)
        Y = randn(rng, 100, 2)
        model = estimate_var(Y, 1)
        n = 2
        L = MEM.safe_cholesky(model.Sigma)
        n_angles = n * (n - 1) ÷ 2

        for dist in [:student_t, :mixture_normal, :mixture_normal_asym, :pml, :skew_normal]
            ndp = MEM._n_dist_params(dist) * n
            angles = zeros(n_angles)
            dp = zeros(ndp)
            ll = MEM._nongaussian_loglik(angles, dp, model.U, L, n; distribution=dist)
            @test isfinite(ll)
            @test ll < 0  # T159: all five densities give negative ll here (observed ≤ -261)

            # ForwardDiff gradient & Hessian match finite differences
            p0 = vcat([0.15], fill(0.25, ndp))
            f_ll = p -> begin
                a, d = MEM._unpack_nongaussian_params(p, n, dist)
                MEM._nongaussian_loglik(a, d, model.U, L, n; distribution=dist)
            end
            g_ad = MEM.ForwardDiff.gradient(f_ll, p0)
            h_fd = 1e-6
            g_fd = [begin
                e = zeros(length(p0)); e[i] = h_fd
                (f_ll(p0 + e) - f_ll(p0 - e)) / (2 * h_fd)
            end for i in eachindex(p0)]
            @test maximum(abs.(g_ad - g_fd)) < 1e-5
            H_ad = MEM.ForwardDiff.hessian(f_ll, p0)
            H_fd = MEM._numerical_hessian(f_ll, p0; eps_step=1e-4)
            @test maximum(abs.(H_ad - H_fd)) < 1e-3
        end
    end

    @testset "Non-Gaussian vcov shape" begin
        rng = Xoshiro(7026)
        Y = randn(rng, 80, 2)
        model = estimate_var(Y, 1)
        n = 2
        L = MEM.safe_cholesky(model.Sigma)
        n_angles = n * (n - 1) ÷ 2
        ndp = MEM._n_dist_params(:student_t) * n
        params = zeros(n_angles + ndp)
        vcov_mat, se_mat = MEM._nongaussian_vcov(params, model.U, L, n, :student_t)
        @test size(vcov_mat) == (n_angles + ndp, n_angles + ndp)
        @test size(se_mat) == (n, n)
        @test all(se_mat .>= 0)
        @test_throws ArgumentError MEM._nongaussian_vcov(params, model.U, L, n, :student_t; vcov_type=:invalid)

        # n == 1 guard (n_angles == 0)
        Y1 = randn(rng, 60, 1)
        m1 = estimate_var(Y1, 1)
        L1 = MEM.safe_cholesky(m1.Sigma)
        v1, se1 = MEM._nongaussian_vcov([0.5], m1.U, L1, 1, :student_t)
        @test size(v1) == (1, 1)
        @test se1 == zeros(1, 1)
    end

    @testset "NGID-01 AC-1: n=3 full Jacobian delta-method se_B0" begin
        rng = Xoshiro(9011)
        B_true3 = [1.0 0.3 -0.2; 0.2 1.1 0.4; -0.1 0.3 0.9]
        Tobs = 300
        eps3 = rand(rng, TDist(5.0), Tobs, 3) .* sqrt(3.0 / 5.0)
        u3 = eps3 * B_true3'
        Y3 = zeros(Tobs, 3)
        for t in 2:Tobs
            Y3[t, :] = 0.4 .* Y3[t - 1, :] .+ u3[t, :]
        end
        m3 = estimate_var(Y3, 1)
        L3 = MEM.safe_cholesky(m3.Sigma)
        angles3 = [0.25, -0.35, 0.15]
        dp3 = fill(log(5.0 - 2.01), 3)
        p3 = vcat(angles3, dp3)

        for vt in (:hessian, :sandwich, :opg)
            vcov_mat, se_mat = MEM._nongaussian_vcov(p3, m3.U, L3, 3, :student_t; vcov_type=vt)
            vcov_angles = vcov_mat[1:3, 1:3]
            J_B = MEM.ForwardDiff.jacobian(
                θ -> vec(Matrix(L3) * MEM._givens_to_orthogonal(θ, 3)), angles3)
            se_ref = reshape(sqrt.(max.(diag(J_B * vcov_angles * J_B'), 0.0)), 3, 3)
            @test maximum(abs.(se_mat - se_ref)) < 1e-10
        end

        # Verify full Jacobian captures off-diagonal angle covariances
        vcov_h, se_h = MEM._nongaussian_vcov(p3, m3.U, L3, 3, :student_t; vcov_type=:hessian)
        vcov_ang = vcov_h[1:3, 1:3]
        @test maximum(abs.(vcov_ang - Diagonal(diag(vcov_ang)))) > 1e-6
        J_B = MEM.ForwardDiff.jacobian(
            θ -> vec(Matrix(L3) * MEM._givens_to_orthogonal(θ, 3)), angles3)
        se_diag_only = reshape(sqrt.(max.(diag(J_B * Diagonal(diag(vcov_ang)) * J_B'), 0.0)), 3, 3)
        @test maximum(abs.(se_h - se_diag_only)) > 1e-4

        # Verify end-to-end identify_student_t result matches _vec_B0_vcov
        res3 = identify_student_t(m3; seed=9011)
        VB_res = MEM._vec_B0_vcov(res3, m3)
        @test VB_res !== nothing
        se_res_ref = reshape(sqrt.(max.(diag(VB_res), 0.0)), 3, 3)
        @test maximum(abs.(res3.se - se_res_ref)) < 1e-10
    end

    @testset "NGID-01 det(Q) < 0 _vec_B0_vcov, _aligned_overid_wald, and vcov sign alignment" begin
        # 1. Student-t(4) reproduction with odd column flip (det(Q) < 0)
        B_d = [-1.0 0.3 -0.2; 0.2 1.1 0.4; -0.1 0.3 0.9]
        rng = Xoshiro(9008)
        ep_t = rand(rng, TDist(4.0), 300, 3) .* sqrt(2.0 / 4.0)
        u_t = ep_t * B_d'
        Y_t = zeros(300, 3)
        for t in 2:300
            Y_t[t, :] = 0.3 .* Y_t[t - 1, :] .+ u_t[t, :]
        end
        md_t = estimate_var(Y_t, 1)
        rs_t = identify_student_t(md_t; seed=8)
        if det(rs_t.Q) > 0
            rs_t = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                label_shocks(rs_t; by=:reference, B_ref=rs_t.B0 * Diagonal([1.0, 1.0, -1.0]))
            end
        end
        @test det(rs_t.Q) < 0

        VB_t = MEM._vec_B0_vcov(rs_t, md_t)
        @test VB_t !== nothing
        Lm_t = Matrix(MEM.safe_cholesky(md_t.Sigma))
        th_t = MEM._Q_givens_angles(rs_t.Q, 3)
        S_last = Diagonal([1.0, 1.0, -1.0])
        J_true_t = MEM.ForwardDiff.jacobian(
            a -> vec(Lm_t * MEM._givens_to_orthogonal(a, 3) * S_last), th_t)
        VB_true_t = J_true_t * rs_t.vcov[1:3, 1:3] * J_true_t'
        @test maximum(abs.(VB_t - VB_true_t)) < 1e-12

        mask = BitMatrix([false true true; true false true; false false false])
        w_got, perm, signs, B_al, _ = MEM._aligned_overid_wald(rs_t, md_t, rs_t.B0, rs_t.se, mask)
        VB_true_al = MEM._align_vec_B0_vcov(VB_true_t, perm, signs, 3)
        idx = [i + (j - 1) * 3 for j in 1:3 for i in 1:3 if mask[i, j]]
        w_true = MEM._wald_rvr(vec(B_al)[idx], VB_true_al[idx, idx])
        @test isapprox(w_got.statistic, w_true.statistic; rtol=1e-10)
        @test isapprox(w_got.pvalue, w_true.pvalue; atol=1e-12)

        # 2. Asymmetric distributions (:pml and :mixture_normal_asym) with det(Q) < 0 and Cov(θ, dp[n]) sign alignment
        rng = Xoshiro(9001)
        ep_p = rand(rng, TDist(4.0), 300, 3) .* sqrt(2.0 / 4.0)
        u_p = ep_p * B_d'
        Y_p = zeros(300, 3)
        for t in 2:300
            Y_p[t, :] = 0.3 .* Y_p[t - 1, :] .+ u_p[t, :]
        end
        md_p = estimate_var(Y_p, 1)

        d_mix = MixtureModel([Normal(-1.0, 0.5), Normal(2.0, 1.2)], [0.65, 0.35])
        rng = Xoshiro(9021)  # well-separated asymmetric mixture for interior MLE + alignment check
        ep_m = (rand(rng, d_mix, 400, 3) .- mean(d_mix)) ./ std(d_mix)
        u_m = ep_m * B_d'
        Y_m = zeros(400, 3)
        for t in 2:400
            Y_m[t, :] = 0.3 .* Y_m[t - 1, :] .+ u_m[t, :]
        end
        md_m = estimate_var(Y_m, 1)

        for (dist, md_a, rs_asym_raw) in ((:pml, md_p, identify_pml(md_p; seed=8)),
                                          (:mixture_normal_asym, md_m, identify_mixture_normal(md_m; asymmetric=true, seed=8)))
            rs_asym = det(rs_asym_raw.Q) < 0 ? rs_asym_raw :
                Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                    label_shocks(rs_asym_raw; by=:reference, B_ref=rs_asym_raw.B0 * S_last)
                end
            @test det(rs_asym.Q) < 0
            Lm_a = Matrix(MEM.safe_cholesky(md_a.Sigma))
            Z_a = Matrix{Float64}(transpose(LowerTriangular(Lm_a) \ transpose(md_a.U)))
            ld_a = -sum(i -> log(abs(Lm_a[i, i])), 1:3)
            VB_a = MEM._vec_B0_vcov(rs_asym, md_a)
            @test VB_a !== nothing
            th_a = MEM._Q_givens_angles(rs_asym.Q, 3)
            J_true_a = MEM.ForwardDiff.jacobian(
                a -> vec(Lm_a * MEM._givens_to_orthogonal(a, 3) * S_last), th_a)
            VB_true_a = J_true_a * rs_asym.vcov[1:3, 1:3] * J_true_a'
            @test maximum(abs.(VB_a - VB_true_a)) < 1e-12

            # Build aligned dp matching rs_asym.dist_params directly
            dp_al = MEM._ml_dist_init(rs_asym, [1, 2, 3])
            p_al = vcat(th_a, dp_al)
            # Evaluate oracle sandwich covariance at Q(a) = _givens_to_orthogonal(a, 3) * S_last
            obs_oracle = (p, z_t) -> begin
                Tθ = eltype(p)
                a_p = p[1:3]
                dp_p = p[4:end]
                Q_p = MEM._givens_to_orthogonal(a_p, 3) * S_last
                eps_t = Q_p' * z_t
                ll = Tθ(ld_a)
                for j in 1:3
                    if dist === :pml
                        idx_j = (j - 1) * 2
                        ll += MEM._pearson_iv_logpdf(eps_t[j], dp_p[idx_j + 1], Tθ(2.05) + exp(dp_p[idx_j + 2]))
                    else
                        idx_j = (j - 1) * 3
                        pm, m1, s1, m2, s2 = MEM._unpack_shock_mixture_asym(dp_p[idx_j + 1], dp_p[idx_j + 2], dp_p[idx_j + 3])
                        ll += MEM._mixture_normal_logpdf(eps_t[j], pm, m1, s1, m2, s2)
                    end
                end
                ll
            end
            H_or = MEM.ForwardDiff.hessian(p -> -sum(t -> obs_oracle(p, @view(Z_a[t, :])), 1:size(Z_a, 1)), p_al)
            H_inv_or = MEM._spd_inv(H_or)
            G_or = reduce(vcat, [MEM.ForwardDiff.gradient(p -> obs_oracle(p, @view(Z_a[t, :])), p_al)' for t in 1:size(Z_a, 1)])
            J_or = (G_or' * G_or + (G_or' * G_or)') / 2
            vcov_or = (H_inv_or * J_or * H_inv_or + (H_inv_or * J_or * H_inv_or)') / 2
            k_flip = dist === :pml ? 3 + 5 : 3 + 8
            @test maximum(abs.(rs_asym.vcov[1:3, k_flip])) > 1e-4  # non-vacuous cross-covariance
            @test maximum(abs.(rs_asym.vcov - vcov_or)) < 1e-10
        end

        # 3. Sibling _vec_B0_vcov methods (GARCHSVARResult & SmoothTransitionSVARResult) with det(Q) < 0
        θ_s = [0.25, -0.15, 0.30]
        Q_pos = MEM._givens_to_orthogonal(θ_s, 3)
        Q_neg = Q_pos * S_last
        B0_neg = Lm_t * Q_neg
        V_ang = [0.02 0.004 -0.003; 0.004 0.03 0.005; -0.003 0.005 0.025]
        V_garch = zeros(12, 12)
        V_garch[1:3, 1:3] = V_ang
        V_garch[4:12, 4:12] = Matrix(0.01I, 9, 9)
        garch_res = GARCHSVARResult{Float64}(B0_neg, Q_neg, ones(3, 3), ones(300, 3), zeros(300, 3),
                                             -100.0, true, 10, fill(0.1, 3, 3), V_garch)
        VB_garch = MEM._vec_B0_vcov(garch_res, md_t)
        @test VB_garch !== nothing
        J_garch_true = MEM.ForwardDiff.jacobian(
            a -> vec(Lm_t * MEM._givens_to_orthogonal(a, 3) * S_last), θ_s)
        @test maximum(abs.(VB_garch - J_garch_true * V_ang * J_garch_true')) < 1e-12

        # SmoothTransitionSVARResult with det(Q) < 0
        n_st = 3
        n_L_st = n_st * (n_st + 1) ÷ 2
        n_ang_st = n_st * (n_st - 1) ÷ 2
        s_var = collect(range(-2.0, 2.0, length=300))
        σs = std(s_var)
        pdim_st = n_L_st + n_ang_st + n_st + 2
        V_st = Matrix(0.005I, pdim_st, pdim_st)
        V_st[n_L_st .+ (1:3), n_L_st .+ (1:3)] = V_ang
        V_st[1, n_L_st + 3] = 0.002
        V_st[n_L_st + 3, 1] = 0.002
        st_res = SmoothTransitionSVARResult{Float64}(
            B0_neg, Q_neg, [md_t.Sigma, 2.0 * md_t.Sigma],
            [ones(3), [1.5, 2.0, 3.0]], 1.0 / σs, 0.0, s_var, fill(0.5, 300),
            -100.0, true, 10, fill(0.1, 3, 3), V_st, md_t.U)
        VB_st = MEM._vec_B0_vcov(st_res, md_t)
        @test VB_st !== nothing
        logγ_lo = log(1e-3 / σs)
        logγ_hi = log(20.0 / σs)
        p_st = vcat(MEM._st_pack_L(Lm_t), θ_s, log.([1.5, 2.0, 3.0]),
                    MEM._st_x_from_gamma(1.0 / σs, logγ_lo, logγ_hi), 0.0)
        J_st_true = MEM.ForwardDiff.jacobian(z -> begin
            Lp = MEM._st_unpack_L(view(z, 1:n_L_st), n_st)
            Qp = MEM._givens_to_orthogonal(z[(n_L_st + 1):(n_L_st + n_ang_st)], n_st) * S_last
            vec(Lp * Qp)
        end, p_st)
        @test maximum(abs.(VB_st - J_st_true * V_st * J_st_true')) < 1e-12
    end

    @testset "NGID-01 label_shocks vcov chart transform, _vec_B0_vcov, and test_overidentification" begin
        # 1. _orthogonal_to_givens ForwardDiff.Dual compatibility (n = 2, 3, 4) & edge cases
        for nn in (2, 3, 4)
            na = nn * (nn - 1) ÷ 2
            θ0 = [0.15 + 0.1 * k for k in 1:na]
            perm_n = nn == 2 ? [2, 1] : (nn == 3 ? [3, 1, 2] : [3, 1, 4, 2])
            signs_n = nn == 2 ? [-1, 1] : (nn == 3 ? [-1, 1, -1] : [-1, 1, 1, -1])
            Q0 = MEM._givens_to_orthogonal(θ0, nn)
            Qn = MEM._col_signed(Q0, perm_n, signs_n)
            M_n = MEM._givens_chart_M(Q0, Qn, perm_n, signs_n, nn)
            θn = MEM._Q_givens_angles(Qn, nn)
            J_ad = MEM.ForwardDiff.jacobian(
                θ -> MEM._orthogonal_to_givens(MEM._givens_to_orthogonal(θ, nn) * M_n, nn), θ0)
            J_new = MEM.ForwardDiff.jacobian(θ -> vec(MEM._givens_to_orthogonal(θ, nn)), θn)
            J_old = MEM.ForwardDiff.jacobian(θ -> vec(MEM._givens_to_orthogonal(θ, nn) * M_n), θ0)
            @test all(isfinite, J_ad)
            @test maximum(abs.(J_new * J_ad - J_old)) < 1e-12
        end
        # ±π and gimbal-lock fallback remain finite
        J_pi = MEM._givens_chart_jacobian([0.0], [-1.0 0.0; 0.0 -1.0], [Float64(π)], 2)
        @test isapprox(J_pi, ones(1, 1); atol=1e-12)
        P_gl = [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0]
        J_gl = MEM._givens_chart_jacobian(zeros(3), P_gl, MEM._Q_givens_angles(P_gl, 3), 3)
        @test size(J_gl) == (3, 3) && all(isfinite, J_gl)

        # Independent oracle covariance at (result.Q, result.dist_params)
        function _oracle_ml_vcov(res::NonGaussianMLResult, mod::VARModel)
            nn = nvars(mod)
            Lm = Matrix(MEM.safe_cholesky(mod.Sigma))
            Zm = Matrix{Float64}(transpose(LowerTriangular(Lm) \ transpose(mod.U)))
            ld = -sum(i -> log(abs(Lm[i, i])), 1:nn)
            θv = MEM._Q_givens_angles(res.Q, nn)
            dpv = MEM._ml_dist_init(res, collect(1:nn))
            pv0 = vcat(θv, dpv)
            Sl = det(res.Q) < 0 ? Diagonal([j < nn ? 1.0 : -1.0 for j in 1:nn]) : I
            d_sym = res.distribution
            obs_ll = (pv, z_t) -> begin
                Tθ = eltype(pv)
                Qp = MEM._givens_to_orthogonal(pv[1:3], 3) * Sl
                eps_t = Qp' * z_t
                ll = Tθ(ld)
                dpp = pv[4:end]
                for j in 1:3
                    if d_sym === :student_t
                        ll += MEM._student_t_logpdf(eps_t[j], Tθ(2.01) + exp(dpp[j]))
                    elseif d_sym === :pml
                        idx = (j - 1) * 2
                        ll += MEM._pearson_iv_logpdf(eps_t[j], dpp[idx + 1], Tθ(2.05) + exp(dpp[idx + 2]))
                    elseif d_sym === :mixture_normal_asym
                        idx = (j - 1) * 3
                        pm, m1, s1, m2, s2 = MEM._unpack_shock_mixture_asym(
                            dpp[idx + 1], dpp[idx + 2], dpp[idx + 3])
                        ll += MEM._mixture_normal_logpdf(eps_t[j], pm, m1, s1, m2, s2)
                    elseif d_sym === :skew_normal
                        ll += MEM._skew_normal_logpdf(eps_t[j], dpp[j])
                    end
                end
                ll
            end
            vt = res.dist_params[:vcov_type]
            if vt === :opg
                G = reduce(vcat, [MEM.ForwardDiff.gradient(pv -> obs_ll(pv, @view(Zm[t, :])), pv0)' for t in 1:size(Zm, 1)])
                J = (G' * G + (G' * G)') / 2
                return Matrix(MEM._spd_inv(J))
            end
            H = MEM.ForwardDiff.hessian(pv -> -sum(t -> obs_ll(pv, @view(Zm[t, :])), 1:size(Zm, 1)), pv0)
            Hinv = MEM._spd_inv(H)
            if vt === :hessian
                return Matrix((Hinv + Hinv') / 2)
            end
            G = reduce(vcat, [MEM.ForwardDiff.gradient(pv -> obs_ll(pv, @view(Zm[t, :])), pv0)' for t in 1:size(Zm, 1)])
            J = (G' * G + (G' * G)') / 2
            V = Hinv * J * Hinv
            Matrix((V + V') / 2)
        end

        B3 = [1.0 0.35 -0.25; 0.20 1.05 0.40; -0.30 0.25 1.10]
        # Student-t DGP
        ep_t3 = rand(Xoshiro(9021), TDist(4.0), 450, 3) .* sqrt(2.0 / 4.0)
        Y_t3 = zeros(450, 3)
        u_t3 = ep_t3 * B3'
        for t in 2:450; Y_t3[t, :] = 0.3 .* Y_t3[t - 1, :] .+ u_t3[t, :]; end
        m_t3 = estimate_var(Y_t3, 1)

        # Asymmetric mixture DGP (for :mixture_normal_asym)
        d_mix3 = MixtureModel([Normal(1.2, 0.6), Normal(-0.4, 0.8)], [0.25, 0.75])
        ep_m3 = (rand(Xoshiro(9019), d_mix3, 400, 3) .- mean(d_mix3)) ./ std(d_mix3)
        Y_m3 = zeros(400, 3)
        u_m3 = ep_m3 * B3'
        for t in 2:400; Y_m3[t, :] = 0.3 .* Y_m3[t - 1, :] .+ u_m3[t, :]; end
        m_m3 = estimate_var(Y_m3, 1)

        # Uncentered skew-normal VARModel (matches uncentered _skew_normal_logpdf 2φ(x)Φ(αx))
        sn_raw = SkewNormal(0.0, 1.0, 3.0)
        ep_sn3 = rand(Xoshiro(9055), sn_raw, 500, 3)
        u_sn3 = ep_sn3 * B3'
        Y_sn3 = zeros(500, 3)
        for t in 2:500; Y_sn3[t, :] = 0.3 .* Y_sn3[t - 1, :] .+ u_sn3[t, :]; end
        m_sn_base = estimate_var(Y_sn3, 1)
        m_sn3 = VARModel(m_sn_base.Y, m_sn_base.p, m_sn_base.B, u_sn3[2:end, :],
                         (u_sn3[2:end, :]' * u_sn3[2:end, :]) / (size(u_sn3, 1) - 1),
                         m_sn_base.aic, m_sn_base.bic, m_sn_base.hqic, m_sn_base.varnames)

        mask_orig = BitMatrix([false true true; false false true; false false false])
        perm_ref = [3, 1, 2]
        signs_ref = [-1, 1, -1]
        # Asymmetric sign restriction matrix pinning perm_flip = [2, 1, 3] (det flip) uniquely
        perm_flip = [2, 1, 3]

        r_cases = (
            (:student_t, m_t3, identify_student_t(m_t3; tol=1e-11, seed=8), 1e-10),
            (:pml, m_t3, identify_pml(m_t3; tol=1e-11, seed=8), 1e-10),
            (:mixture_normal_asym, m_m3, identify_mixture_normal(m_m3; asymmetric=true, vcov_type=:opg, seed=8), 1e-10),
            (:mixture_normal_asym, m_m3, identify_mixture_normal(m_m3; asymmetric=true, tol=1e-11, max_iter=1000, seed=8), 1e-8),
            (:skew_normal, m_sn3, identify_skew_normal(m_sn3; tol=1e-11, seed=8), 1e-10),
        )
        for (d_name, m_d, r_d, or_tol) in r_cases
            # 2a. by=:reference (non-trivial perm + sign flips, det preserved)
            r_ref = label_shocks(r_d; by=:reference, B_ref=r_d.B0[:, perm_ref] .* signs_ref')
            @test det(r_ref.Q) * det(r_d.Q) > 0
            # 2b. by=:restrictions & odd-permutation by=:reference (flips det(Q): +1 -> -1 and -1 -> +1)
            S_rest = Int.(sign.(r_d.B0[:, perm_flip]))
            r_rest = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                label_shocks(r_d; by=:restrictions, restrictions=S_rest)
            end
            p_rest = [argmax([abs(dot(r_rest.Q[:, j], r_d.Q[:, k])) for k in 1:3]) for j in 1:3]
            @test sort(p_rest) == [1, 2, 3]
            r_flip = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                label_shocks(r_d; by=:reference, B_ref=r_d.B0[:, perm_flip])
            end
            @test det(r_flip.Q) * det(r_d.Q) < 0

            ov_orig = test_overidentification(m_d, r_d; restrictions=mask_orig)
            @test haskey(ov_orig.details, :wald_statistic)

            for (r_lab, p_lab) in ((r_ref, perm_ref), (r_rest, p_rest), (r_flip, perm_flip))
                VB_lab = MEM._vec_B0_vcov(r_lab, m_d)
                @test VB_lab !== nothing
                @test maximum(abs.(reshape(sqrt.(diag(VB_lab)), 3, 3) - r_lab.se)) < 1e-10

                V_or = _oracle_ml_vcov(r_lab, m_d)
                @test maximum(abs.(r_lab.vcov - V_or)) < or_tol

                if haskey(r_lab.dist_params, :alpha)
                    @test r_lab.dist_params[:all_params] == r_lab.dist_params[:alpha]
                elseif haskey(r_lab.dist_params, :kappa)
                    @test r_lab.dist_params[:all_params] ==
                          vcat(r_lab.dist_params[:kappa], r_lab.dist_params[:nu])
                elseif haskey(r_lab.dist_params, :mu1)
                    @test r_lab.dist_params[:all_params] ==
                          vcat(r_lab.dist_params[:p_mix], r_lab.dist_params[:mu1], r_lab.dist_params[:sigma1])
                end

                # Same-order mask vs permuted mask invariance
                mask_perm = mask_orig[:, p_lab]
                ov_perm = test_overidentification(m_d, r_lab; restrictions=mask_perm)
                @test isapprox(ov_perm.statistic, ov_orig.statistic; atol=1e-8)
                @test isapprox(ov_perm.details[:wald_statistic], ov_orig.details[:wald_statistic]; atol=1e-8)

                ov_same = test_overidentification(m_d, r_lab; restrictions=mask_orig)
                @test isapprox(ov_same.statistic, ov_orig.statistic; atol=1e-8)
                @test isapprox(ov_same.details[:wald_statistic], ov_orig.details[:wald_statistic]; atol=1e-8)
            end
        end

        # 3. Empty / dummy vcov guards (must not throw)
        r_empty = NonGaussianMLResult{Float64}(
            B3, Matrix{Float64}(I, 3, 3), zeros(10, 3), :student_t, -1.0, -2.0,
            Dict{Symbol,Any}(:nu => [4.0, 5.0, 6.0], :all_params => [4.0, 5.0, 6.0]),
            zeros(0, 0), zeros(3, 3), true, 1, 1.0, 2.0)
        @test isempty(label_shocks(r_empty; by=:reference, B_ref=B3[:, perm_ref]).vcov)

        # 4. Sibling result types under label_shocks (det preserved & det flipped)
        Lm_s = Matrix(MEM.safe_cholesky(m_t3.Sigma))
        θ_sib = [0.25, -0.15, 0.30]
        Q_sib = MEM._givens_to_orthogonal(θ_sib, 3)
        B0_sib = Lm_s * Q_sib
        V_ang_sib = [0.02 0.004 -0.003; 0.004 0.03 0.005; -0.003 0.005 0.025]
        se_sib, _ = MEM._delta_B0_se(θ_sib, V_ang_sib, a -> Lm_s * MEM._givens_to_orthogonal(a, 3), 3)

        V_garch_s = Matrix(0.01I, 12, 12)
        V_garch_s[1:3, 1:3] = V_ang_sib
        V_garch_s[1, 5] = 0.002; V_garch_s[5, 1] = 0.002
        garch_s = GARCHSVARResult{Float64}(
            B0_sib, Q_sib, ones(3, 3), ones(450, 3) .+ 0.2, zeros(450, 3),
            -100.0, true, 10, se_sib, V_garch_s)
        for (B_ref_s, pm_s) in ((B0_sib[:, perm_ref] .* signs_ref', perm_ref), (B0_sib[:, perm_flip], perm_flip))
            g_l = label_shocks(garch_s; by=:reference, B_ref=B_ref_s)
            VB_g = MEM._vec_B0_vcov(g_l, m_t3)
            @test VB_g !== nothing
            @test maximum(abs.(reshape(sqrt.(diag(VB_g)), 3, 3) - g_l.se)) < 1e-12
            w_g1 = MEM._aligned_overid_wald(garch_s, m_t3, garch_s.B0, garch_s.se, mask_orig)[1]
            w_g2 = MEM._aligned_overid_wald(g_l, m_t3, g_l.B0, g_l.se, mask_orig[:, pm_s])[1]
            @test isapprox(w_g1.statistic, w_g2.statistic; atol=1e-10)
        end

        gmm_s = NonGaussianGMMResult{Float64}(
            B0_sib, Q_sib, θ_sib, V_ang_sib, se_sib, 1.0, 0.5, :both, :two_step,
            zeros(450, 3), m_t3.varnames, ["Shock $j" for j in 1:3])
        gmm_pres = label_shocks(gmm_s; by=:reference, B_ref=B0_sib[:, perm_ref] .* signs_ref')
        gmm_flip = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            label_shocks(gmm_s; by=:reference, B_ref=B0_sib[:, perm_flip])
        end
        @test maximum(abs.(reshape(sqrt.(diag(MEM._vec_B0_vcov(gmm_pres, m_t3))), 3, 3) - gmm_pres.se)) < 1e-12
        @test maximum(abs.(reshape(sqrt.(diag(MEM._vec_B0_vcov(gmm_flip, m_t3))), 3, 3) - gmm_flip.se)) < 1e-12

        n_L_s = 6; n_ang_s = 3; pdim_s = n_L_s + n_ang_s + 3 + 2
        s_v = collect(range(-2.0, 2.0, length=450))
        σ_s = std(s_v)
        V_st_s = Matrix(0.005I, pdim_s, pdim_s)
        V_st_s[(n_L_s + 1):(n_L_s + 3), (n_L_s + 1):(n_L_s + 3)] = V_ang_sib
        V_st_s[1, n_L_s + 2] = 0.001; V_st_s[n_L_s + 2, 1] = 0.001
        V_st_s[n_L_s + 2, n_L_s + 4] = -0.0015; V_st_s[n_L_s + 4, n_L_s + 2] = -0.0015
        logγ_lo_s = log(1e-3 / σ_s); logγ_hi_s = log(20.0 / σ_s)
        p_st_s = vcat(MEM._st_pack_L(Lm_s), θ_sib, log.([1.5, 2.5, 3.5]),
                      MEM._st_x_from_gamma(1.0 / σ_s, logγ_lo_s, logγ_hi_s), 0.0)
        se_st_s, _ = MEM._delta_B0_se(
            p_st_s, V_st_s,
            z -> MEM._st_unpack_L(view(z, 1:n_L_s), 3) * MEM._givens_to_orthogonal(z[(n_L_s + 1):(n_L_s + 3)], 3), 3)
        st_s = SmoothTransitionSVARResult{Float64}(
            B0_sib, Q_sib, [m_t3.Sigma, 2.0 * m_t3.Sigma], [ones(3), [1.5, 2.5, 3.5]],
            1.0 / σ_s, 0.0, s_v, fill(0.5, 450), -100.0, true, 10, se_st_s, V_st_s, m_t3.U)
        st_flip = label_shocks(st_s; by=:reference, B_ref=B0_sib[:, perm_flip])
        @test maximum(abs.(reshape(sqrt.(diag(MEM._vec_B0_vcov(st_flip, m_t3))), 3, 3) - st_flip.se)) < 1e-12
        @test isapprox(test_lambda_distinct(st_s).statistic[[1, 3, 2]],
                       test_lambda_distinct(st_flip).statistic; atol=1e-12)

        V_ms_s = Matrix(0.01I, 12, 12)
        V_ms_s[1, 10] = 0.002; V_ms_s[10, 1] = 0.002
        se_ms_s = reshape(sqrt.(diag(V_ms_s[1:9, 1:9])), 3, 3)
        ms_s = MarkovSwitchingSVARResult{Float64}(
            B0_sib, Q_sib, [m_t3.Sigma, 2.0 * m_t3.Sigma], [ones(3), [1.5, 2.5, 3.5]],
            fill(0.5, 450, 2), [0.9 0.1; 0.1 0.9], -100.0, true, 10, 2, se_ms_s, V_ms_s, 0.9, zeros(450, 3))
        ms_l = label_shocks(ms_s; by=:reference, B_ref=B0_sib[:, perm_ref] .* signs_ref')
        @test maximum(abs.(reshape(sqrt.(diag(MEM._vec_B0_vcov(ms_l, m_t3))), 3, 3) - ms_l.se)) < 1e-12
        w_ms1 = MEM._aligned_overid_wald(ms_s, m_t3, ms_s.B0, ms_s.se, mask_orig)[1]
        w_ms2 = MEM._aligned_overid_wald(ms_l, m_t3, ms_l.B0, ms_l.se, mask_orig[:, perm_ref])[1]
        @test isapprox(w_ms1.statistic, w_ms2.statistic; atol=1e-12)

        @testset "NGID-01 regression: q < n_angles, LewisTVVResult, VAR(p=2), n=4/5, Float32, BigFloat" begin
        # 1. Direct gimbal-lock mask[1,1]==true vs mask[1,2]==true vs mask[1,3]==true (n=3, q=1)
        m11 = BitMatrix([true false false; false false false; false false false])
        m12 = BitMatrix([false true false; false false false; false false false])
        m13 = BitMatrix([false false true; false false false; false false false])
        m_q2 = BitMatrix([false true true; false false false; false false false])

        for (idx_rc, (d_name, m_d, r_d, _)) in enumerate(r_cases)
            idx_rc == 3 && continue  # avoid duplicating :mixture_normal_asym (same B0/Q, different vcov_type)
            # Direct invariance without label_shocks (gimbal lock check at (1,1))
            ov11 = test_overidentification(m_d, r_d; restrictions=m11)
            ov12 = test_overidentification(m_d, r_d; restrictions=m12)
            ov13 = test_overidentification(m_d, r_d; restrictions=m13)
            @test isapprox(ov11.statistic, ov12.statistic; rtol=1e-5, atol=1e-6)
            @test isapprox(ov12.statistic, ov13.statistic; rtol=1e-5, atol=1e-6)
            @test isapprox(ov11.details[:wald_statistic], ov12.details[:wald_statistic]; rtol=1e-5, atol=1e-6)

            # label_shocks (det preserved via :reference & det flipped via :restrictions) for q=1, 2
            r_ref = label_shocks(r_d; by=:reference, B_ref=r_d.B0[:, perm_ref] .* signs_ref')
            S_rest = Int.(sign.(r_d.B0[:, perm_flip]))
            r_rest = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                label_shocks(r_d; by=:restrictions, restrictions=S_rest)
            end
            p_rest = [argmax([abs(dot(r_rest.Q[:, j], r_d.Q[:, k])) for k in 1:3]) for j in 1:3]

            for qmask in (FAST ? (m12,) : (m12, m_q2))
                ov0 = test_overidentification(m_d, r_d; restrictions=qmask)
                for (r_lab, p_lab) in ((r_ref, perm_ref), (r_rest, p_rest))
                    ov_p = test_overidentification(m_d, r_lab; restrictions=qmask[:, p_lab])
                    ov_s = test_overidentification(m_d, r_lab; restrictions=qmask)
                    @test isapprox(ov_p.statistic, ov0.statistic; rtol=1e-5, atol=1e-6)
                    @test isapprox(ov_s.statistic, ov0.statistic; rtol=1e-5, atol=1e-6)
                    @test isapprox(ov_p.details[:wald_statistic], ov0.details[:wald_statistic]; rtol=1e-5, atol=1e-6)
                    @test isapprox(ov_s.details[:wald_statistic], ov0.details[:wald_statistic]; rtol=1e-5, atol=1e-6)
                end
            end
        end

        # 2. LewisTVVResult _vec_B0_vcov, test_overidentification, & 2-col-flipped identity label_shocks
        tvv_s = LewisTVVResult{Float64}(
            B0_sib, Q_sib, θ_sib, V_ang_sib, se_sib, 2.5, 0.4, 3, [1, 2, 3], :two_step,
            true, 10, false, 5.0, "converged", zeros(450, 3), m_t3.varnames, ["Shock $j" for j in 1:3])
        tvv_pres = label_shocks(tvv_s; by=:reference, B_ref=B0_sib[:, perm_ref] .* signs_ref')
        tvv_flip = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            label_shocks(tvv_s; by=:reference, B_ref=B0_sib[:, perm_flip])
        end
        @test det(tvv_pres.Q) > 0 && det(tvv_flip.Q) < 0 && all(isnan, tvv_flip.theta)
        for rt in (tvv_s, tvv_pres, tvv_flip)
            VB_tvv = MEM._vec_B0_vcov(rt, m_t3)
            @test VB_tvv !== nothing
            @test maximum(abs.(reshape(sqrt.(diag(VB_tvv)), 3, 3) - rt.se)) < 1e-12
        end
        ov_tvv_j = test_overidentification(m_t3, tvv_s)
        @test ov_tvv_j.details[:method] === :hansen_j
        @test ov_tvv_j.details[:lags] == [1, 2, 3]
        @test isapprox(ov_tvv_j.statistic, 2.5)
        ov_tvv0 = test_overidentification(m_t3, tvv_s; restrictions=mask_orig)
        ov_tvvp = test_overidentification(m_t3, tvv_pres; restrictions=mask_orig[:, perm_ref])
        ov_tvvf = test_overidentification(m_t3, tvv_flip; restrictions=mask_orig[:, perm_flip])
        @test isapprox(ov_tvv0.statistic, ov_tvvp.statistic; atol=1e-10)
        @test isapprox(ov_tvv0.statistic, ov_tvvf.statistic; atol=1e-10)

        # Identity label_shocks when 2 columns were flipped (Q != Q_raw while det(Q) > 0)
        θ_2flip = [2.8, 0.2, 0.3]
        Q_2flip_raw = MEM._givens_to_orthogonal(θ_2flip, 3)
        D_2flip = Diagonal([-1.0, -1.0, 1.0])
        Q_2flip = Q_2flip_raw * D_2flip
        B0_2flip = Lm_s * Q_2flip
        se_2flip, _ = MEM._delta_B0_se(θ_2flip, V_ang_sib,
                                       a -> Lm_s * MEM._givens_to_orthogonal(a, 3) * D_2flip, 3)
        gmm_2f = NonGaussianGMMResult{Float64}(
            B0_2flip, Q_2flip, θ_2flip, V_ang_sib, se_2flip, 1.0, 0.5, :both, :two_step,
            zeros(450, 3), m_t3.varnames, ["Shock $j" for j in 1:3])
        gmm_2f_id = label_shocks(gmm_2f; by=:reference, B_ref=B0_2flip)
        @test maximum(abs.(reshape(sqrt.(diag(MEM._vec_B0_vcov(gmm_2f_id, m_t3))), 3, 3) - gmm_2f_id.se)) < 1e-12
        tvv_2f = LewisTVVResult{Float64}(
            B0_2flip, Q_2flip, θ_2flip, V_ang_sib, se_2flip, 2.5, 0.4, 3, [1, 2, 3], :two_step,
            true, 10, false, 5.0, "converged", zeros(450, 3), m_t3.varnames, ["Shock $j" for j in 1:3])
        tvv_2f_id = label_shocks(tvv_2f; by=:reference, B_ref=B0_2flip)
        @test maximum(abs.(reshape(sqrt.(diag(MEM._vec_B0_vcov(tvv_2f_id, m_t3))), 3, 3) - tvv_2f_id.se)) < 1e-12

        # 3. Non-ML test_overidentification at q < n_angles (n=3, q=1, 2; det preserved & flipped)
        B_ev = [1.0 0.35 -0.25; 0.20 1.05 0.40; -0.30 0.25 1.10]
        ep_ev = vcat(randn(Xoshiro(9881), 150, 3), randn(Xoshiro(9882), 150, 3) .* [1.0 2.0 3.0])
        Y_ev = zeros(300, 3); u_ev = ep_ev * B_ev'
        for t in 2:300; Y_ev[t, :] = 0.2 .* Y_ev[t - 1, :] .+ u_ev[t, :]; end
        m_ev = estimate_var(Y_ev, 1)
        s_ev = vcat(fill(-1.0, 150), fill(1.0, 149)) .+ 0.2 .* randn(Xoshiro(9883), size(m_ev.U, 1))
        rg_ev = identify_garch(m_ev; max_iter=50)
        @test maximum(abs.(reshape(sqrt.(diag(MEM._vec_B0_vcov(rg_ev, m_ev))), 3, 3) - rg_ev.se)) < 1e-12
        nonml_cases = (
            identify_external_volatility(m_ev, vcat(fill(1, 150), fill(2, 149)); regimes=2),
            identify_markov_switching(m_ev; n_regimes=2, n_starts=1, max_iter=25, rng=Xoshiro(9884)),
            identify_smooth_transition(m_ev, s_ev),
            rg_ev,
            gmm_s,
        )
        for r_nm in nonml_cases
            m_nm = (r_nm === gmm_s) ? m_t3 : m_ev
            r_nm_ref = label_shocks(r_nm; by=:reference, B_ref=r_nm.B0[:, perm_ref] .* signs_ref')
            r_nm_flip = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                label_shocks(r_nm; by=:reference, B_ref=r_nm.B0[:, perm_flip])
            end
            for qmask in (FAST ? (m12,) : (m12, m_q2))
                o0 = test_overidentification(m_nm, r_nm; restrictions=qmask)
                or = test_overidentification(m_nm, r_nm_ref; restrictions=qmask[:, perm_ref])
                of = test_overidentification(m_nm, r_nm_flip; restrictions=qmask[:, perm_flip])
                @test isapprox(o0.statistic, or.statistic; rtol=1e-5, atol=1e-6)
                @test isapprox(o0.statistic, of.statistic; rtol=1e-5, atol=1e-6)
                if haskey(o0.details, :wald_statistic)
                    @test isapprox(o0.details[:wald_statistic], or.details[:wald_statistic]; rtol=1e-5, atol=1e-6)
                    @test isapprox(o0.details[:wald_statistic], of.details[:wald_statistic]; rtol=1e-5, atol=1e-6)
                end
            end
        end

        # 4. VAR(p=2) across all 5 ML distributions (label_shocks, _vec_B0_vcov, q=2 & q=3)
        ep_p2 = rand(Xoshiro(9022), TDist(4.0), 300, 3) .* sqrt(2.0 / 4.0)
        Y_p2 = zeros(302, 3); u_p2 = vcat(zeros(2, 3), ep_p2) * B_ev'
        for t in 3:302; Y_p2[t, :] = 0.25 .* Y_p2[t - 1, :] .- 0.10 .* Y_p2[t - 2, :] .+ u_p2[t, :]; end
        m_p2 = estimate_var(Y_p2, 2)
        ep_p2_sn = rand(Xoshiro(9023), SkewNormal(0.0, 1.0, 3.0), 300, 3)
        Y_p2_sn = zeros(302, 3); u_p2_sn = vcat(zeros(2, 3), ep_p2_sn) * B_ev'
        for t in 3:302; Y_p2_sn[t, :] = 0.25 .* Y_p2_sn[t - 1, :] .- 0.10 .* Y_p2_sn[t - 2, :] .+ u_p2_sn[t, :]; end
        m_p2b = estimate_var(Y_p2_sn, 2)
        m_p2_sn = VARModel(m_p2b.Y, m_p2b.p, m_p2b.B, u_p2_sn[3:end, :],
                           (u_p2_sn[3:end, :]' * u_p2_sn[3:end, :]) / (size(u_p2_sn, 1) - 2),
                           m_p2b.aic, m_p2b.bic, m_p2b.hqic, m_p2b.varnames)
        p2_cases = FAST ? ((:student_t, m_p2), (:mixture_normal_asym, m_p2)) :
                          ((:student_t, m_p2), (:pml, m_p2), (:mixture_normal, m_p2),
                           (:mixture_normal_asym, m_p2), (:skew_normal, m_p2_sn))
        for (dist_p2, md_p2) in p2_cases
            rp2 = identify_nongaussian_ml(md_p2; distribution=dist_p2, n_starts=1, max_iter=120, seed=8)
            rp2_ref = label_shocks(rp2; by=:reference, B_ref=rp2.B0[:, perm_ref] .* signs_ref')
            VB_p2 = MEM._vec_B0_vcov(rp2_ref, md_p2)
            @test VB_p2 !== nothing
            @test maximum(abs.(reshape(sqrt.(diag(VB_p2)), 3, 3) - rp2_ref.se)) < 1e-10
            for qmask in (m_q2, mask_orig)
                op1 = test_overidentification(md_p2, rp2; restrictions=qmask)
                op2 = test_overidentification(md_p2, rp2_ref; restrictions=qmask[:, perm_ref])
                @test isapprox(op1.statistic, op2.statistic; rtol=1e-5, atol=1e-6)
                @test isapprox(op1.details[:wald_statistic], op2.details[:wald_statistic]; rtol=1e-5, atol=1e-6)
            end
        end

        # 5. n = 4 (q = 2 < n_angles = 6) & n = 5 (q = 2 < n_angles = 10)
        B4 = [1.0 0.3 -0.2 0.15; 0.2 1.05 0.35 -0.1; -0.25 0.2 1.1 0.25; 0.1 -0.15 0.2 0.95]
        ep4 = rand(Xoshiro(9401), TDist(4.0), 300, 4) .* sqrt(2.0 / 4.0)
        Y4 = zeros(300, 4); u4 = ep4 * B4'
        for t in 2:300; Y4[t, :] = 0.25 .* Y4[t - 1, :] .+ u4[t, :]; end
        m4 = estimate_var(Y4, 1)
        perm4 = [3, 1, 4, 2]; signs4 = [-1, 1, 1, -1]
        mask4_q2 = BitMatrix([false true true false; false false false false;
                              false false false false; false false false false])
        for dist4 in (FAST ? (:student_t, :pml) : (:student_t, :pml, :mixture_normal_asym))
            r4 = identify_nongaussian_ml(m4; distribution=dist4, n_starts=1, max_iter=120, seed=12)
            r4_ref = label_shocks(r4; by=:reference, B_ref=r4.B0[:, perm4] .* signs4')
            VB4 = MEM._vec_B0_vcov(r4_ref, m4)
            @test VB4 !== nothing
            @test maximum(abs.(reshape(sqrt.(diag(VB4)), 4, 4) - r4_ref.se)) < 1e-10
            ov4_0 = test_overidentification(m4, r4; restrictions=mask4_q2)
            ov4_r = test_overidentification(m4, r4_ref; restrictions=mask4_q2[:, perm4])
            @test isapprox(ov4_0.statistic, ov4_r.statistic; rtol=1e-5, atol=1e-6)
            @test isapprox(ov4_0.details[:wald_statistic], ov4_r.details[:wald_statistic]; rtol=1e-5, atol=1e-6)
        end

        B5 = [1.0 0.2 -0.1 0.15 -0.1; 0.15 1.0 0.25 -0.1 0.2; -0.2 0.1 1.0 0.2 -0.15;
              0.1 -0.15 0.2 1.0 0.25; -0.1 0.2 -0.1 0.15 1.0]
        ep5 = rand(Xoshiro(9501), TDist(4.0), 300, 5) .* sqrt(2.0 / 4.0)
        Y5 = zeros(300, 5); u5 = ep5 * B5'
        for t in 2:300; Y5[t, :] = 0.25 .* Y5[t - 1, :] .+ u5[t, :]; end
        m5 = estimate_var(Y5, 1)
        perm5 = [3, 1, 5, 2, 4]; signs5 = [-1, 1, -1, 1, 1]
        mask5_q2 = falses(5, 5); mask5_q2[1, 2] = true; mask5_q2[2, 3] = true
        for dist5 in (FAST ? (:student_t,) : (:student_t, :mixture_normal))
            r5 = identify_nongaussian_ml(m5; distribution=dist5, n_starts=1, max_iter=120, seed=11)
            r5_ref = label_shocks(r5; by=:reference, B_ref=r5.B0[:, perm5] .* signs5')
            VB5 = MEM._vec_B0_vcov(r5_ref, m5)
            @test VB5 !== nothing
            @test maximum(abs.(reshape(sqrt.(diag(VB5)), 5, 5) - r5_ref.se)) < 1e-10
            ov5_0 = test_overidentification(m5, r5; restrictions=mask5_q2)
            ov5_r = test_overidentification(m5, r5_ref; restrictions=mask5_q2[:, perm5])
            @test isapprox(ov5_0.statistic, ov5_r.statistic; rtol=1e-5, atol=1e-6)
            @test isapprox(ov5_0.details[:wald_statistic], ov5_r.details[:wald_statistic]; rtol=1e-5, atol=1e-6)
        end

        # 6. Float32 end-to-end & _re_loggamma Float32/BigFloat edge checks
        m32 = VARModel(Float32.(m_t3.Y), m_t3.p, Float32.(m_t3.B), Float32.(m_t3.U),
                       Float32.(m_t3.Sigma), Float32(m_t3.aic), Float32(m_t3.bic),
                       Float32(m_t3.hqic), m_t3.varnames)
        m32_sn = VARModel(Float32.(m_sn3.Y), m_sn3.p, Float32.(m_sn3.B), Float32.(m_sn3.U),
                        Float32.(m_sn3.Sigma), Float32(m_sn3.aic), Float32(m_sn3.bic),
                        Float32(m_sn3.hqic), m_sn3.varnames)
        @test identify_fastica(m32; approach=:symmetric, seed=7) isa ICASVARResult{Float32}
        f32_cases = FAST ? ((:student_t, m32), (:skew_normal, m32_sn)) :
                           ((:student_t, m32), (:pml, m32), (:mixture_normal, m32),
                            (:mixture_normal_asym, m32), (:skew_normal, m32_sn))
        for (dist32, md32) in f32_cases
            r32 = identify_nongaussian_ml(md32; distribution=dist32, n_starts=1, max_iter=120, seed=7)
            @test r32 isa NonGaussianMLResult{Float32}
            @test all(isfinite, r32.se)
            r32_ref = label_shocks(r32; by=:reference, B_ref=r32.B0[:, perm_ref] .* Float32.(signs_ref'))
            VB32 = MEM._vec_B0_vcov(r32_ref, md32)
            @test VB32 !== nothing
            @test maximum(abs.(reshape(sqrt.(diag(VB32)), 3, 3) - r32_ref.se)) < 1e-5
            ov32_1 = test_overidentification(md32, r32; restrictions=mask_orig)
            ov32_2 = test_overidentification(md32, r32_ref; restrictions=mask_orig[:, perm_ref])
            @test isapprox(ov32_1.details[:wald_statistic], ov32_2.details[:wald_statistic]; rtol=1e-4, atol=1e-4)
            @test isapprox(ov32_1.statistic, ov32_2.statistic; atol=5e-2)
        end
        @test MEM._re_loggamma(3.0f0, 2.0f0) isa Float32
        @test isapprox(MEM._re_loggamma(big(3.0), big(2.0)), big(MEM._re_loggamma(3.0, 2.0)); atol=1e-12)
        end
    end

    # =========================================================================
    # Heteroskedastic Internals (src/nongaussian/heteroskedastic.jl)
    # =========================================================================

    @testset "Eigendecomposition identification" begin
        rng = Xoshiro(7030)
        n = 3
        # Create two distinct covariance matrices
        A = randn(rng, n, n)
        Sigma1 = A * A' + I
        B = randn(rng, n, n)
        Sigma2 = B * B' + I

        B0, Q, Lambda = MEM._eigendecomposition_id(Sigma1, Sigma2)
        @test size(B0) == (n, n)
        @test size(Q) == (n, n)
        @test length(Lambda) == n
        # B0 * B0' should approximate Sigma1
        @test norm(B0 * B0' - Sigma1) / norm(Sigma1) < 0.5
    end

    @testset "Hamilton filter" begin
        rng = Xoshiro(7031)
        T_obs = 100
        n = 2
        U = randn(rng, T_obs, n)
        Sigma1 = U' * U / T_obs + I
        Sigma2 = 2 * U' * U / T_obs + I
        P = [0.9 0.1; 0.1 0.9]

        filtered, predicted, loglik = MEM._hamilton_filter(U, [Sigma1, Sigma2], P)
        @test size(filtered) == (T_obs, 2)
        @test size(predicted) == (T_obs, 2)
        @test isfinite(loglik)
        @test loglik < 0  # T159: mixture loglik on randn data (observed -153.4)
        # Probabilities should sum to ~1
        @test all(isapprox.(sum(filtered, dims=2), 1.0, atol=1e-6))
    end

    @testset "Hamilton smoother" begin
        rng = Xoshiro(7032)
        T_obs = 50
        n = 2
        U = randn(rng, T_obs, n)
        Sigma1 = U' * U / T_obs + I
        Sigma2 = 2 * U' * U / T_obs + I
        P = [0.9 0.1; 0.1 0.9]

        filtered, predicted, _ = MEM._hamilton_filter(U, [Sigma1, Sigma2], P)
        smoothed = MEM._hamilton_smoother(filtered, predicted, P)
        @test size(smoothed) == (T_obs, 2)
        # Probabilities should be valid
        @test all(smoothed .>= 0)
        @test all(isapprox.(sum(smoothed, dims=2), 1.0, atol=1e-4))
    end

    @testset "MS EM step" begin
        rng = Xoshiro(7033)
        T_obs = 50
        n = 2
        K = 2
        U = randn(rng, T_obs, n)
        # Build proper filtered/predicted/smoothed probabilities via Hamilton filter
        Sigma1 = U' * U / T_obs + I
        Sigma2 = 2 * U' * U / T_obs + I
        P = [0.9 0.1; 0.1 0.9]
        filtered, predicted, _ = MEM._hamilton_filter(U, [Sigma1, Sigma2], P)
        smoothed = MEM._hamilton_smoother(filtered, predicted, P)

        Sigma_new, P_new = MEM._ms_em_step(U, smoothed, filtered, predicted, P, K)
        @test length(Sigma_new) == K
        @test all(size(S) == (n, n) for S in Sigma_new)
        # Covariance matrices should be PSD
        for S in Sigma_new
            @test all(eigvals(Symmetric(S)) .> -1e-10)
        end
        # Transition matrix rows sum to 1
        @test all(isapprox.(sum(P_new, dims=2), 1.0, atol=1e-10))
    end

    @testset "MS regime order is increasing trace" begin
        Σq = [1.0 0.2; 0.2 1.0]
        Σl = [4.0 0.5; 0.5 5.0]
        P = [0.8 0.2; 0.3 0.7]
        sm = [0.1 0.9; 0.2 0.8; 0.15 0.85]
        Σo, Po, smo = MEM._ms_order_regimes([Σl, Σq], P, sm)
        @test tr(Σo[1]) < tr(Σo[2])
        @test Σo[1] == Σq && Σo[2] == Σl
        @test Po == [0.7 0.3; 0.2 0.8]
        @test smo == sm[:, [2, 1]]
        Σs, _, _ = MEM._ms_order_regimes([Σq, Σl], P, sm)
        @test Σs[1] === Σq
    end

    @testset "GARCH(1,1) filter" begin
        rng = Xoshiro(7034)
        eps_sq = abs2.(randn(rng, 100))
        h = MEM._garch11_filter(0.01, 0.05, 0.9, eps_sq)
        @test length(h) == 100
        @test all(h .> 0)
        # h[1] should be unconditional variance
        @test h[1] ≈ 0.01 / max(1.0 - 0.05 - 0.9, eps())
    end

    @testset "GARCH(1,1) loglik" begin
        rng = Xoshiro(7035)
        eps_sq = abs2.(randn(rng, 100))
        # Valid params: omega ~ 0.01, alpha ~ 0.12, beta ~ 0.5
        # sigmoid(-2) ≈ 0.12 → alpha ≈ 0.06, sigmoid(0) = 0.5 → beta ≈ 0.495
        params = [log(0.01), -2.0, 0.0]
        ll = MEM._garch11_loglik(params, eps_sq)
        # T159: kept — value is DGP-scale-dependent (observed +482.2, densities > 1); the isinf-on-nonstationary below is the real pin.
        @test isfinite(ll)

        # alpha + beta >= 1 should return Inf (non-stationary)
        params_bad = [log(0.01), 5.0, 5.0]
        ll_bad = MEM._garch11_loglik(params_bad, eps_sq)
        @test isinf(ll_bad)
    end

    @testset "Estimate GARCH(1,1)" begin
        rng = Xoshiro(7036)
        # Use simpler data: squared normal (no explosive dynamics)
        eps_sq = abs2.(randn(rng, 200))
        omega, alpha, beta, h_est = MEM._estimate_garch11(eps_sq)
        @test omega > 0
        @test alpha >= 0
        @test beta >= 0
        @test length(h_est) == 200
        @test all(h_est .> 0)
    end

    @testset "Logistic transition" begin
        # At threshold: G = 0.5
        @test MEM._logistic_transition(0.0, 1.0, 0.0) ≈ 0.5
        # Far above threshold: G ≈ 1
        @test MEM._logistic_transition(10.0, 1.0, 0.0) > 0.99
        # Far below threshold: G ≈ 0
        @test MEM._logistic_transition(-10.0, 1.0, 0.0) < 0.01
        # High gamma: sharp transition
        @test MEM._logistic_transition(0.01, 100.0, 0.0) > 0.5
        @test MEM._logistic_transition(-0.01, 100.0, 0.0) < 0.5
    end

    # =========================================================================
    # Tests Internals (src/nongaussian/tests.jl)
    # =========================================================================

    @testset "Permutations" begin
        p1 = MEM._permutations(1)
        @test p1 == [[1]]
        p2 = MEM._permutations(2)
        @test length(p2) == 2
        @test Set(p2) == Set([[1, 2], [2, 1]])
        p3 = MEM._permutations(3)
        @test length(p3) == 6  # 3!
    end

    @testset "Procrustes distance" begin
        # Identity: distance to itself is 0
        B = [1.0 0.0; 0.0 1.0]
        @test MEM._procrustes_distance(B, B) ≈ 0.0 atol = 1e-10
        # Signed permutation: distance still 0
        B_perm = [0.0 -1.0; 1.0 0.0]
        # This is a rotation, not a signed permutation, but signed perms include columns swaps with signs
        B2 = [0.0 1.0; -1.0 0.0]
        @test MEM._procrustes_distance(B, [1.0 0.0; 0.0 -1.0]) ≈ 0.0 atol = 1e-10
    end

    @testset "Cross-correlation test" begin
        rng = Xoshiro(7040)
        # Independent shocks
        shocks = randn(rng, 100, 3)
        stat, pval, df = MEM._cross_correlation_test(shocks, 5)
        @test stat >= 0
        @test 0 <= pval <= 1
        @test df == 3 * (3 - 1) ÷ 2 * (5 + 1)  # 18
    end

    @testset "dCov independence test" begin
        rng = Xoshiro(7041)
        # Independent shocks
        shocks = randn(rng, 50, 2)
        stat, pval = MEM._dcov_independence_test(shocks)
        @test stat >= 0
        @test 0 <= pval <= 1
    end

    @testset "#566 Pearson IV is a genuine normalized density" begin
        # ν = 0 reduces exactly to the unit-variance scaled Student t with 2m−1 df
        for m in (3.0, 8.0)
            @test maximum(abs(MEM._pearson_iv_logpdf(x, 0.0, m) -
                              MEM._student_t_logpdf(x, 2m - 1)) for x in -6:0.5:6) < 1e-10
        end
        # ∫f = 1, mean 0, variance 1 at skewed parameters (sinh-grid quadrature).
        # The pre-#566 cubic-tilt pseudo-density diverged for every κ ≠ 0.
        for (v, m) in ((2.0, 3.0), (-4.0, 5.0), (1.0, 2.2))
            ts = range(-15.0, 15.0; length=200_001)
            dt = step(ts)
            Z = m1 = m2 = 0.0
            for t in ts
                x = sinh(t)
                w = exp(MEM._pearson_iv_logpdf(x, v, m)) * cosh(t) * dt
                Z += w; m1 += x * w; m2 += x^2 * w
            end
            @test isapprox(Z, 1.0; atol=1e-6)
            @test isapprox(m1, 0.0; atol=1e-6)
            @test isapprox(m2, 1.0; atol=1e-5)
        end
        # finite everywhere the optimizer can reach
        # T159: kept — optimizer-domain smoke; the Z/m1/m2 ≈ pins above guard the pdf.
        @test all(isfinite, [MEM._pearson_iv_logpdf(x, v, m)
                             for x in (-30.0, 0.0, 30.0), v in (-60.0, 0.0, 60.0),
                                 m in (2.05, 5.0, 9912.0)])
    end
end

# =============================================================================
# T090 (#189) SUB-2: ICA buffer reuse is arithmetic-identical (exact)
# =============================================================================

@testset "T090 SUB-2: buffered dcov/HSIC == allocating formulas" begin
    rng = Xoshiro(19002)
    n = 40
    x = randn(rng, n)
    y = x .^ 2 .+ 0.3 .* randn(rng, n)

    # Pre-change distance covariance formula, inlined
    A = [abs(x[i] - x[j]) for i in 1:n, j in 1:n]
    B = [abs(y[i] - y[j]) for i in 1:n, j in 1:n]
    A_row = mean(A, dims=2); A_col = mean(A, dims=1); A_grand = mean(A)
    B_row = mean(B, dims=2); B_col = mean(B, dims=1); B_grand = mean(B)
    for i in 1:n, j in 1:n
        A[i, j] = A[i, j] - A_row[i] - A_col[j] + A_grand
        B[i, j] = B[i, j] - B_row[i] - B_col[j] + B_grand
    end
    dcov_ref = max(sum(A .* B) / n^2, 0.0)
    @test MEM._distance_covariance(x, y) == dcov_ref  # bit-identical

    # Pre-change HSIC formula, inlined
    s2 = 2 * 1.0^2
    K = [exp(-(x[i] - x[j])^2 / s2) for i in 1:n, j in 1:n]
    L = [exp(-(y[i] - y[j])^2 / s2) for i in 1:n, j in 1:n]
    H = I - ones(n, n) / n
    Kc = H * K * H
    Lc = H * L * H
    hsic_ref = tr(Kc * Lc) / (n - 1)^2
    @test MEM._hsic_statistic(x, y; sigma=1.0) == hsic_ref  # bit-identical

    # Buffered objective == unbuffered objective (same buffers semantics)
    Z = randn(rng, 60, 2)
    angles = [0.3]
    @test MEM._dcov_objective(angles, Z, 2) ==
          MEM._dcov_objective(angles, Z, 2,
                              [Matrix{Float64}(undef, 60, 60) for _ in 1:3]...)
    Hbuf = I - ones(60, 60) / 60
    bufs = ntuple(_ -> Matrix{Float64}(undef, 60, 60), 6)
    @test MEM._hsic_objective(angles, Z, 2; sigma=1.0) ==
          MEM._hsic_objective(angles, Z, 2, bufs..., Hbuf; sigma=1.0)
end

@testset "SID-22 Holm adjustment and shock helpers" begin
    p = [0.01, 0.04, 0.03]
    h = MEM._holm_adjust(p)
    # Sorted p: 0.01, 0.03, 0.04. Holm: 3·0.01=0.03; max(0.03, 2·0.03)=0.06;
    # max(0.06, 1·0.04)=0.06.
    @test h[1] ≈ 0.03
    @test h[3] ≈ 0.06
    @test h[2] ≈ 0.06
    @test all(0 .<= h .<= 1)

    B = [1.0 0.25; 0.4 1.2]
    perm, signs = MEM._match_columns(B, B)
    @test perm == [1, 2]
    @test all(==(1), signs)

    shocks = randn(Xoshiro(75190), 40, 2)
    dummy = ICASVARResult{Float64}(B, Matrix{Float64}(I, 2, 2),
                                   Matrix{Float64}(I, 2, 2), shocks,
                                   :fastica, true, 1, 0.0)
    @test MEM._result_shocks(dummy) == shocks
    @test MEM._result_method(dummy) == :fastica

    B0 = [1.0 0.2; 0.3 1.1]
    se = [0.1 0.05; 0.08 0.12]
    mask = BitMatrix([false true; true false])
    V = zeros(4, 4)
    for j in 1:2, i in 1:2
        V[i + (j - 1) * 2, i + (j - 1) * 2] = se[i, j]^2
    end
    w_diag = MEM._wald_B0_zeros(B0, se, mask)
    w_rvr = MEM._wald_B0_zeros(B0, se, mask; vcov_B=V)
    @test w_diag.approximation == :independence
    @test w_rvr.approximation == :rvr
    @test w_rvr.statistic ≈ w_diag.statistic atol=1e-10
    V2 = copy(V)
    i12 = 1 + 1 * 2
    i21 = 2 + 0 * 2
    V2[i12, i21] = V2[i21, i12] = 0.3 * se[1, 2] * se[2, 1]
    w_c = MEM._wald_B0_zeros(B0, se, mask; vcov_B=V2)
    @test w_c.approximation == :rvr
    @test !isapprox(w_c.statistic, w_diag.statistic; rtol=1e-3)
end

@testset "LowerTriangular backing store (Julia 1.13)" begin
    # Julia 1.13 returns cholesky().L as LowerTriangular{T,Adjoint}; the ML
    # loglikelihood must accept any backing store.
    rng = Xoshiro(113115)
    U = randn(rng, 50, 2)
    S = Symmetric(cov(U) + 0.1 * I)
    Lm = MEM.safe_cholesky(Matrix(S))
    La = LowerTriangular(Matrix(Matrix(Lm)')')
    @test La isa LowerTriangular{Float64,Adjoint{Float64,Matrix{Float64}}}
    ll_m = MEM._nongaussian_loglik([0.3], [1.0, 1.0], U, Lm, 2; distribution=:student_t)
    ll_a = MEM._nongaussian_loglik([0.3], [1.0, 1.0], U, La, 2; distribution=:student_t)
    @test isfinite(ll_m) && ll_a ≈ ll_m
end
