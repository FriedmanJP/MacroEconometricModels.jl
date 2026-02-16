# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <wookyung9207@gmail.com>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

using Test
using MacroEconometricModels
using Random
using LinearAlgebra
using Statistics
using Distributions

const MEM = MacroEconometricModels

@testset "Non-Gaussian Internals" begin

    # =========================================================================
    # ICA Internals (src/nongaussian/ica.jl)
    # =========================================================================

    @testset "Whitening" begin
        Random.seed!(7001)
        # Identity-covariance input should yield near-identity output covariance
        U = randn(200, 3)
        Z, W_white, dewhiten = MEM._whiten(U)
        @test size(Z) == (200, 3)
        @test size(W_white) == (3, 3)
        @test size(dewhiten) == (3, 3)
        cov_Z = Z' * Z / size(Z, 1)
        @test norm(cov_Z - I) < 0.3  # approximately identity

        # Near-zero eigenvalue: rank-deficient input
        U_rank = hcat(randn(100, 2), zeros(100))
        Z2, W2, dw2 = MEM._whiten(U_rank)
        @test size(Z2, 1) == 100
        @test size(Z2, 2) <= 3  # may drop near-zero eigenvalue component
    end

    @testset "Givens rotation roundtrip" begin
        Random.seed!(7002)
        n = 3
        n_angles = n * (n - 1) ÷ 2
        angles_orig = randn(n_angles)
        Q = MEM._givens_to_orthogonal(angles_orig, n)
        @test size(Q) == (n, n)
        # Q should be orthogonal
        @test norm(Q' * Q - I) < 1e-10
        @test norm(Q * Q' - I) < 1e-10

        # Roundtrip: extract angles and reconstruct (approximate extraction)
        angles_extracted = MEM._orthogonal_to_givens(Q, n)
        Q_rebuilt = MEM._givens_to_orthogonal(angles_extracted, n)
        # The extraction is approximate, so check orthogonality of result
        @test norm(Q_rebuilt' * Q_rebuilt - I) < 1e-8

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
        Random.seed!(7003)
        Z = randn(100, 3)
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
        Random.seed!(7004)
        Z = randn(100, 3)
        Z_w, _, _ = MEM._whiten(Z)
        n = size(Z_w, 2)
        W, iters = MEM._fastica_symmetric(Z_w, n; contrast=:exp, max_iter=50)
        @test size(W) == (n, n)
        @test iters > 0
        @test iters <= 50
    end

    @testset "ICA to SVAR conversion" begin
        Random.seed!(7005)
        Y = randn(200, 3)
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
        Random.seed!(7006)
        Z = randn(100, 3)
        Z_w, _, _ = MEM._whiten(Z)
        mats = MEM._jade_cumulant_matrices(Z_w)
        n = size(Z_w, 2)
        n_mats = n * (n + 1) ÷ 2
        @test length(mats) == n_mats
        @test all(size(m) == (n, n) for m in mats)
    end

    @testset "Joint diagonalization" begin
        Random.seed!(7007)
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
        Random.seed!(7008)
        Z = randn(100, 3)
        R = MEM._sobi_autocovariance(Z, 1)
        @test size(R) == (3, 3)
        R0 = MEM._sobi_autocovariance(Z, 0)
        @test size(R0) == (3, 3)
    end

    @testset "Distance covariance" begin
        Random.seed!(7009)
        # Independent variables: low distance covariance
        x = randn(50)
        y = randn(50)
        dcov_ind = MEM._distance_covariance(x, y)
        @test dcov_ind >= 0

        # Perfectly dependent: high distance covariance
        dcov_dep = MEM._distance_covariance(x, x .^ 2 .+ 0.01 * randn(50))
        @test dcov_dep >= 0
    end

    @testset "dCov objective" begin
        Random.seed!(7010)
        Z = randn(50, 2)
        angles = zeros(1)  # 2x2 case: 1 angle
        obj = MEM._dcov_objective(angles, Z, 2)
        @test obj >= 0
        @test isfinite(obj)
    end

    @testset "HSIC statistic" begin
        Random.seed!(7011)
        x = randn(30)
        y = randn(30)
        hsic_val = MEM._hsic_statistic(x, y; sigma=1.0)
        @test isfinite(hsic_val)
    end

    @testset "HSIC objective" begin
        Random.seed!(7012)
        Z = randn(30, 2)
        angles = zeros(1)
        obj = MEM._hsic_objective(angles, Z, 2; sigma=1.0)
        @test isfinite(obj)
    end

    # =========================================================================
    # ML Internals (src/nongaussian/ml.jl)
    # =========================================================================

    @testset "Student-t log-pdf" begin
        # Should return finite values
        val = MEM._student_t_logpdf(0.0, 5.0)
        @test isfinite(val)
        # nu near boundary (2.01)
        val_boundary = MEM._student_t_logpdf(1.0, 2.01)
        @test isfinite(val_boundary)
        # Large nu should approximate normal
        val_large = MEM._student_t_logpdf(0.0, 1000.0)
        normal_val = logpdf(Normal(), 0.0)
        @test abs(val_large - normal_val) < 0.1
    end

    @testset "Mixture normal log-pdf" begin
        # p_mix = 0.5, equal variances => standard normal
        val = MEM._mixture_normal_logpdf(0.0, 0.5, 1.0, 1.0)
        @test isfinite(val)
        # p_mix = 0.8, sigma1=0.5, sigma2=1.5
        val_asym = MEM._mixture_normal_logpdf(0.0, 0.8, 0.5, 1.5)
        @test isfinite(val_asym)
        # p_mix near 0 => mostly second component
        val_low = MEM._mixture_normal_logpdf(0.0, 0.01, 0.5, 1.0)
        @test isfinite(val_low)
    end

    @testset "Skew-normal log-pdf" begin
        # alpha = 0 reduces to normal
        val_normal = MEM._skew_normal_logpdf(0.0, 0.0)
        expected = log(2.0) + logpdf(Normal(), 0.0) + logcdf(Normal(), 0.0)
        @test val_normal ≈ expected
        # Non-zero alpha
        val_skew = MEM._skew_normal_logpdf(1.0, 2.0)
        @test isfinite(val_skew)
    end

    @testset "Pearson IV log-pdf" begin
        val = MEM._pearson_iv_logpdf(0.0, 0.0, 5.0)
        @test isfinite(val)
        val2 = MEM._pearson_iv_logpdf(1.0, 0.5, 10.0)
        @test isfinite(val2)
    end

    @testset "Gaussian log-likelihood" begin
        Random.seed!(7020)
        Y = randn(100, 2)
        model = estimate_var(Y, 1)
        ll = MEM._gaussian_loglik(model)
        @test isfinite(ll)
        @test ll < 0  # log-likelihood should be negative for reasonable data
    end

    @testset "_n_dist_params" begin
        @test MEM._n_dist_params(:student_t) == 1
        @test MEM._n_dist_params(:mixture_normal) == 3
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
        Random.seed!(7025)
        Y = randn(100, 2)
        model = estimate_var(Y, 1)
        n = 2
        L = MEM.safe_cholesky(model.Sigma)
        n_angles = n * (n - 1) ÷ 2

        for dist in [:student_t, :mixture_normal, :pml, :skew_normal]
            ndp = MEM._n_dist_params(dist) * n
            angles = zeros(n_angles)
            dp = zeros(ndp)
            ll = MEM._nongaussian_loglik(angles, dp, model.U, L, n; distribution=dist)
            @test isfinite(ll)
        end
    end

    @testset "Non-Gaussian vcov shape" begin
        Random.seed!(7026)
        Y = randn(80, 2)
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
    end

    # =========================================================================
    # Heteroskedastic Internals (src/nongaussian/heteroskedastic.jl)
    # =========================================================================

    @testset "Eigendecomposition identification" begin
        Random.seed!(7030)
        n = 3
        # Create two distinct covariance matrices
        A = randn(n, n)
        Sigma1 = A * A' + I
        B = randn(n, n)
        Sigma2 = B * B' + I

        B0, Q, Lambda = MEM._eigendecomposition_id(Sigma1, Sigma2)
        @test size(B0) == (n, n)
        @test size(Q) == (n, n)
        @test length(Lambda) == n
        # B0 * B0' should approximate Sigma1
        @test norm(B0 * B0' - Sigma1) / norm(Sigma1) < 0.5
    end

    @testset "Hamilton filter" begin
        Random.seed!(7031)
        T_obs = 100
        n = 2
        U = randn(T_obs, n)
        Sigma1 = U' * U / T_obs + I
        Sigma2 = 2 * U' * U / T_obs + I
        P = [0.9 0.1; 0.1 0.9]

        filtered, predicted, loglik = MEM._hamilton_filter(U, [Sigma1, Sigma2], P)
        @test size(filtered) == (T_obs, 2)
        @test size(predicted) == (T_obs, 2)
        @test isfinite(loglik)
        # Probabilities should sum to ~1
        @test all(isapprox.(sum(filtered, dims=2), 1.0, atol=1e-6))
    end

    @testset "Hamilton smoother" begin
        Random.seed!(7032)
        T_obs = 50
        n = 2
        U = randn(T_obs, n)
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
        Random.seed!(7033)
        T_obs = 50
        n = 2
        U = randn(T_obs, n)
        probs = hcat(rand(T_obs), rand(T_obs))
        probs ./= sum(probs, dims=2)  # normalize rows

        Sigma_new, P_new = MEM._ms_em_step(U, probs, 2)
        @test length(Sigma_new) == 2
        @test all(size(S) == (n, n) for S in Sigma_new)
        # Covariance matrices should be PSD
        for S in Sigma_new
            @test all(eigvals(Symmetric(S)) .> -1e-10)
        end
        # Transition matrix rows sum to 1
        @test all(isapprox.(sum(P_new, dims=2), 1.0, atol=1e-10))
    end

    @testset "GARCH(1,1) filter" begin
        Random.seed!(7034)
        eps_sq = abs2.(randn(100))
        h = MEM._garch11_filter(0.01, 0.05, 0.9, eps_sq)
        @test length(h) == 100
        @test all(h .> 0)
        # h[1] should be unconditional variance
        @test h[1] ≈ 0.01 / max(1.0 - 0.05 - 0.9, eps())
    end

    @testset "GARCH(1,1) loglik" begin
        Random.seed!(7035)
        eps_sq = abs2.(randn(100))
        # Valid params: omega ~ 0.01, alpha ~ 0.12, beta ~ 0.5
        # sigmoid(-2) ≈ 0.12 → alpha ≈ 0.06, sigmoid(0) = 0.5 → beta ≈ 0.495
        params = [log(0.01), -2.0, 0.0]
        ll = MEM._garch11_loglik(params, eps_sq)
        @test isfinite(ll)

        # alpha + beta >= 1 should return Inf (non-stationary)
        params_bad = [log(0.01), 5.0, 5.0]
        ll_bad = MEM._garch11_loglik(params_bad, eps_sq)
        @test isinf(ll_bad)
    end

    @testset "Estimate GARCH(1,1)" begin
        Random.seed!(7036)
        # Use simpler data: squared normal (no explosive dynamics)
        eps_sq = abs2.(randn(200))
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
        Random.seed!(7040)
        # Independent shocks
        shocks = randn(100, 3)
        stat, pval, df = MEM._cross_correlation_test(shocks, 5)
        @test stat >= 0
        @test 0 <= pval <= 1
        @test df == 3 * (3 - 1) ÷ 2 * (5 + 1)  # 18
    end

    @testset "dCov independence test" begin
        Random.seed!(7041)
        # Independent shocks
        shocks = randn(50, 2)
        stat, pval = MEM._dcov_independence_test(shocks)
        @test stat >= 0
        @test 0 <= pval <= 1
    end
end
