# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics
using Random
using Distributions: loggamma

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "BVAR Bayesian Parameter Recovery" begin
    _tprint("Generating Data for Bayesian Verification...")

    # 1. Reference DGP (DGP-03 #792): non-diagonal A, non-identity B0, burn-in.
    # b_vecs layout is vec(B) = [c1, A11, A12, c2, A21, A22] (B = [c'; A']).
    rng = MersenneTwister(42)
    T = 100
    n = 2
    p = 1

    true_A = [0.5 0.1; 0.0 0.4]
    B0_true = [1.0 0.0; 0.3 1.0]
    Y = dgp_var(rng; A=true_A, B0=B0_true, T=T).Y

    # 2. Direct Sampler Parameter Recovery (Primary Test)
    @testset "Direct Sampler Parameter Recovery" begin
        _tprint("Estimating BVAR (direct)...")
        post = estimate_bvar(Y, p; n_draws=(FAST ? 30 : 100), sampler=:direct, rng=rng)
        @test post isa BVARPosterior

        # Extract and check parameter recovery
        b_vecs, _ = MacroEconometricModels.extract_chain_parameters(post)
        means_arr = vec(mean(b_vecs, dims=1))

        _tprint("Recovered Means: ", means_arr)

        # Check intercepts (should be near 0)
        @test abs(means_arr[1]) < 0.5
        @test abs(means_arr[4]) < 0.5

        # Check A elements against the truth (keep the draw counts)
        @test isapprox(means_arr[2], 0.5, atol=0.35)  # A11
        @test isapprox(means_arr[3], 0.1, atol=0.35)  # A12
        @test isapprox(means_arr[5], 0.0, atol=0.35)  # A21
        @test isapprox(means_arr[6], 0.4, atol=0.35)  # A22

        _tprint("Direct Sampler Parameter Recovery Verified.")
    end

    # 3. Gibbs Sampler Smoke Test
    @testset "Gibbs Sampler Smoke Test" begin
        _tprint("Estimating BVAR (Gibbs)...")
        post_gibbs = estimate_bvar(Y, p;
            n_draws=(FAST ? 20 : 50), sampler=:gibbs, burnin=(FAST ? 20 : 50), thin=1,
            rng=rng
        )
        @test post_gibbs isa BVARPosterior
        @test post_gibbs.n_draws == (FAST ? 20 : 50)
        @test post_gibbs.sampler == :gibbs
        _tprint("Gibbs Sampler Smoke Test Passed.")
    end

    # DGP-03 #792: posterior calibration on known truth. One n_draws=1000
    # posterior serves four checks: B-draw ESS (a collapsed sampler fails),
    # empirical-quantile vs normal-theory width (wrong scale fails), 70%-band
    # frequentist coverage over 12 fresh datasets (miscalibrated variance
    # fails), and nodal IRF coverage against the var_irf truth (biased IRFs
    # fail). NOTE on indexing: b_vecs = vec(B) with B = [c'; A'], so slope
    # indices [2,3,5,6] are [A11, A12, A21, A22] — the transpose interleaves.
    @testset "Posterior calibration on known truth" begin
        Yc = dgp_var(MersenneTwister(4700); A=true_A, B0=B0_true, T=200).Y
        post_c = estimate_bvar(Yc, 1; n_draws=1000, sampler=:direct,
                               rng=MersenneTwister(4701))

        # B-draw ESS ≥ 200 everywhere (realized min 995: iid direct draws).
        for j in axes(post_c.B_draws, 2), k in axes(post_c.B_draws, 3)
            x = post_c.B_draws[:, j, k]
            r1 = cor(x[1:end-1], x[2:end])
            ess = isfinite(r1) ? 1000 * (1 - r1) / (1 + r1) : 1000
            @test ess >= 200
        end

        # Empirical 16-84 half-width vs normal-theory sd (realized 0.97-0.99;
        # NIW marginals are near-normal at T=200).
        for j in axes(post_c.B_draws, 2), k in axes(post_c.B_draws, 3)
            x = post_c.B_draws[:, j, k]
            q = quantile(x, [0.16, 0.84])
            @test 0.7 <= ((q[2] - q[1]) / 2) / std(x) <= 1.4
        end

        # 70%-band frequentist coverage: 12 fresh datasets × 4 slopes
        # (realized 32/48 = 0.67 vs nominal 0.7; MC SE ≈ 0.07).
        slope_truth = [true_A[1, 1], true_A[1, 2], true_A[2, 1], true_A[2, 2]]
        covers = 0
        for s in 1:12
            Ys = dgp_var(MersenneTwister(9000 + s); A=true_A, B0=B0_true, T=200).Y
            ps = estimate_bvar(Ys, 1; n_draws=500, sampler=:direct,
                               rng=MersenneTwister(9100 + s))
            bv, _ = MacroEconometricModels.extract_chain_parameters(ps)
            for (idx, tr) in zip([2, 3, 5, 6], slope_truth)
                qq = quantile(bv[:, idx], [0.15, 0.85])
                covers += (qq[1] <= tr <= qq[2])
            end
        end
        @test 0.5 <= covers / 48 <= 0.9

        # Nodal IRF coverage against the var_irf truth (realized 0.90 with 90%
        # bands over 40 nodes).
        bir = irf(post_c, 10; method=:cholesky, quantiles=[0.05, 0.5, 0.95])
        TH = var_irf([true_A], cholesky(Symmetric(B0_true * B0_true')).L, 10)
        lo, hi = bir.quantiles[:, :, :, 1], bir.quantiles[:, :, :, 3]
        @test sum(lo[h, i, j] <= TH[h, i, j] <= hi[h, i, j]
                  for h in 1:10, i in 1:n, j in 1:n) / 40 >= 0.7
    end

    # DGP-03 #792: one-step posterior-mean forecasts vs the truth-implied MSE
    # (innovation variances). Fit once on the first 400 of T=800, roll 400
    # origins (realized ratios 1.10/0.91 — estimation penalty plus luck).
    @testset "Forecast MSE within 20% of truth-implied" begin
        Yf = dgp_var(MersenneTwister(4800); A=true_A, B0=B0_true, T=800).Y
        post_f = estimate_bvar(Yf[1:400, :], 1; n_draws=500, sampler=:direct,
                               rng=MersenneTwister(4801))
        Bm = dropdims(mean(post_f.B_draws; dims=1); dims=1)
        Stru = B0_true * B0_true'
        for j in 1:n
            mse = sum((Yf[t+1, j] - dot(vcat(1.0, Yf[t, :]), Bm[:, j]))^2
                      for t in 400:799) / 400
            @test mse <= 1.2 * Stru[j, j]
        end
    end

    # ==========================================================================
    # Robustness Tests
    # ==========================================================================

    @testset "Reproducibility" begin
        _tprint("Testing BVAR reproducibility...")
        rng = MersenneTwister(77777)  # DGP-03: explicit rng
        Y_rep = dgp_var(rng; A=0.5 * Matrix{Float64}(I, 2, 2),
                        B0=Matrix{Float64}(I, 2, 2), T=80).Y

        post1 = estimate_bvar(Y_rep, 1; n_draws=50, sampler=:direct,
                              rng=MersenneTwister(88888))

        post2 = estimate_bvar(Y_rep, 1; n_draws=50, sampler=:direct,
                              rng=MersenneTwister(88888))

        # Same random seed should give same results
        @test post1.B_draws ≈ post2.B_draws
        @test post1.Sigma_draws ≈ post2.Sigma_draws
        _tprint("Reproducibility test passed.")
    end

    @testset "Numerical Stability - Near-Collinear Data" begin
        _tprint("Testing numerical stability with near-collinear data...")
        rng = MersenneTwister(11111)  # DGP-03: explicit rng
        T_nc = 80
        n_nc = 3

        # Create data with near-collinearity
        Y_nc = randn(rng, T_nc, n_nc)
        Y_nc[:, 3] = Y_nc[:, 1] + 0.01 * randn(rng, T_nc)

        post_nc = estimate_bvar(Y_nc, 1; n_draws=50, sampler=:direct, rng=rng)
        @test post_nc isa BVARPosterior

        # Check all parameters are finite
        @test all(isfinite.(post_nc.B_draws))
        _tprint("Numerical stability test passed.")
    end

    @testset "Edge Cases" begin
        _tprint("Testing edge cases...")
        rng = MersenneTwister(22222)  # DGP-03: explicit rng

        # Single variable BVAR
        Y_single = randn(rng, 80, 1)
        post_single = estimate_bvar(Y_single, 1; n_draws=50, rng=rng)
        @test post_single isa BVARPosterior

        # Verify parameter dimensions for single variable
        # k = 1 + n*p = 1 + 1*1 = 2
        @test size(post_single.B_draws, 2) == 2  # intercept + 1 AR coefficient
        @test size(post_single.B_draws, 3) == 1  # 1 variable
        _tprint("Edge case tests passed.")
    end

    @testset "Posterior Draws Structure" begin
        _tprint("Testing posterior draws structure...")
        rng = MersenneTwister(33333)  # DGP-03: explicit rng
        Y_diag = dgp_var(rng; A=0.5 * Matrix{Float64}(I, 2, 2),
                         B0=Matrix{Float64}(I, 2, 2), T=80).Y

        post_diag = estimate_bvar(Y_diag, 1; n_draws=50, sampler=:direct, rng=rng)

        # Check structure
        @test post_diag.n_draws == 50
        @test post_diag.p == 1
        @test post_diag.n == 2

        # All samples should be finite
        @test all(isfinite.(post_diag.B_draws))
        @test all(isfinite.(post_diag.Sigma_draws))

        # Sigma draws should be symmetric positive definite
        for s in 1:post_diag.n_draws
            S = post_diag.Sigma_draws[s, :, :]
            @test isapprox(S, S', atol=1e-10)
            @test all(eigvals(Symmetric(S)) .> -1e-10)
        end

        # Posterior mean should be reasonable (not extreme)
        b_vecs, _ = MacroEconometricModels.extract_chain_parameters(post_diag)
        mean_b = vec(mean(b_vecs, dims=1))
        @test all(abs.(mean_b) .< 10.0)  # Not exploding
        _tprint("Posterior draws structure test passed.")
    end

    @testset "Posterior Model Extraction" begin
        _tprint("Testing posterior model extraction...")
        rng = MersenneTwister(44444)  # DGP-03: explicit rng
        Y_post = dgp_var(rng; A=0.5 * Matrix{Float64}(I, 2, 2),
                         B0=Matrix{Float64}(I, 2, 2), T=80).Y

        post = estimate_bvar(Y_post, 1; n_draws=50, rng=rng)

        # Extract posterior mean model
        mean_model = posterior_mean_model(post; data=Y_post)
        @test mean_model isa VARModel
        @test all(isfinite.(mean_model.B))
        @test all(isfinite.(mean_model.Sigma))

        # Extract posterior median model
        med_model = posterior_median_model(post; data=Y_post)
        @test med_model isa VARModel
        @test all(isfinite.(med_model.B))

        # Test deprecated wrapper signatures
        mean_model2 = posterior_mean_model(post, 1, 2; data=Y_post)
        @test mean_model2 isa VARModel

        _tprint("Posterior model extraction test passed.")
    end

    @testset "Minnesota prior with BVAR" begin
        rng = MersenneTwister(99887)  # DGP-03: explicit rng
        Y_mn = randn(rng, 80, 2)
        hyper = MinnesotaHyperparameters(tau=0.2, decay=2.0, omega=0.5)
        post_mn = estimate_bvar(Y_mn, 1; prior=:minnesota, hyper=hyper, n_draws=100, rng=rng)
        @test post_mn isa BVARPosterior
        @test post_mn.prior == :minnesota
        _tprint("Minnesota prior BVAR test passed.")
    end

    @testset "BVAR sampler variants" begin
        rng = MersenneTwister(99886)  # DGP-03: explicit rng
        Y_sv = randn(rng, 60, 2)

        # Direct sampler
        @testset "Direct sampler" begin
            post_direct = estimate_bvar(Y_sv, 1; sampler=:direct, n_draws=50, rng=rng)
            @test post_direct isa BVARPosterior
            @test post_direct.sampler == :direct
            _tprint("Direct sampler test passed.")
        end

        # Gibbs sampler
        @testset "Gibbs sampler" begin
            post_gibbs = estimate_bvar(Y_sv, 1; sampler=:gibbs, n_draws=50, burnin=100, rng=rng)
            @test post_gibbs isa BVARPosterior
            @test post_gibbs.sampler == :gibbs
            _tprint("Gibbs sampler test passed.")
        end

        # Unknown sampler
        @testset "Unknown sampler error" begin
            @test_throws ArgumentError estimate_bvar(Y_sv, 1; sampler=:nonexistent, n_draws=50)
        end
    end

    # 8. BVARPosterior show() method
    @testset "BVARPosterior show method" begin
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 30 : 50), sampler=:direct, rng=rng)
        io = IOBuffer()
        show(io, post)
        out = String(take!(io))
        @test length(out) > 0
        @test occursin("Bayesian VAR", out)
        @test occursin("Mean", out)
        @test occursin("2.5%", out)
        @test occursin("97.5%", out)
        @test occursin("Posterior Mean", out)
        _tprint("BVARPosterior show test passed.")
    end

    # ==========================================================================
    # Additional Coverage Tests
    # ==========================================================================

    @testset "forecast(BVARPosterior, h)" begin
        rng = MersenneTwister(50001)  # DGP-03: explicit rng
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 30 : 80), sampler=:direct, rng=rng)

        # Basic forecast
        fc = forecast(post, 4)
        @test fc isa BVARForecast
        @test fc.horizon == 4
        @test size(fc.forecast) == (4, 2)
        @test size(fc.ci_lower) == (4, 2)
        @test size(fc.ci_upper) == (4, 2)
        @test all(isfinite.(fc.forecast))
        @test all(fc.ci_lower .<= fc.forecast)
        @test all(fc.forecast .<= fc.ci_upper)
        @test fc.point_estimate == :mean  # default
        @test fc.conf_level == 0.95

        # point_estimate=:mean
        fc_mean = forecast(post, 4; point_estimate=:mean)
        @test fc_mean isa BVARForecast
        @test fc_mean.point_estimate == :mean
        @test all(isfinite.(fc_mean.forecast))

        # Negative horizon error
        @test_throws ArgumentError forecast(post, 0)
        @test_throws ArgumentError forecast(post, -1)

        # Custom conf_level and reps
        fc_90 = forecast(post, 3; conf_level=0.90, reps=10)
        @test fc_90 isa BVARForecast
        @test fc_90.conf_level == Float64(0.90)
        @test fc_90.horizon == 3

        _tprint("forecast(BVARPosterior, h) tests passed.")
    end

    @testset "BVARForecast show method" begin
        rng = MersenneTwister(50002)  # DGP-03: explicit rng
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 30 : 60), sampler=:direct, rng=rng)

        # Show with :median (explicit)
        fc_med = forecast(post, 3; point_estimate=:median)
        io = IOBuffer()
        show(io, fc_med)
        out_med = String(take!(io))
        @test length(out_med) > 0
        @test occursin("Bayesian VAR Forecast", out_med)
        @test occursin("Horizon", out_med)
        @test occursin("Post. Median", out_med)

        # Show with :mean
        fc_mn = forecast(post, 3; point_estimate=:mean)
        io2 = IOBuffer()
        show(io2, fc_mn)
        out_mn = String(take!(io2))
        @test occursin("Post. Mean", out_mn)
        @test occursin("Credibility", out_mn)

        _tprint("BVARForecast show method tests passed.")
    end

    @testset "BVARPosterior show with varnames" begin
        rng = MersenneTwister(50003)  # DGP-03: explicit rng
        post_vn = estimate_bvar(Y, 1; n_draws=(FAST ? 30 : 50), sampler=:direct, rng=rng,
                                varnames=["GDP", "Inflation"])
        io = IOBuffer()
        show(io, post_vn)
        out = String(take!(io))
        @test occursin("GDP", out)
        @test occursin("Inflation", out)

        # Verify varnames stored correctly
        @test post_vn.varnames == ["GDP", "Inflation"]

        _tprint("BVARPosterior show with varnames test passed.")
    end

    @testset "posterior_mean_model and posterior_median_model (default data)" begin
        rng = MersenneTwister(50004)  # DGP-03: explicit rng
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 30 : 50), sampler=:direct, rng=rng)

        # Without explicit data kwarg — should use post.data
        mean_m = posterior_mean_model(post)
        @test mean_m isa VARModel
        @test all(isfinite.(mean_m.B))
        @test all(isfinite.(mean_m.Sigma))

        med_m = posterior_median_model(post)
        @test med_m isa VARModel
        @test all(isfinite.(med_m.B))
        @test all(isfinite.(med_m.Sigma))

        # Mean and median should generally differ (but both valid)
        @test size(mean_m.B) == size(med_m.B)
        @test size(mean_m.Sigma) == size(med_m.Sigma)

        _tprint("posterior_mean_model / posterior_median_model (default data) tests passed.")
    end

    @testset "Deprecated wrapper process_posterior_samples(post, p, n, func)" begin
        rng = MersenneTwister(50005)  # DGP-03: explicit rng
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 20 : 40), sampler=:direct, rng=rng)

        # The 4-arg deprecated wrapper should delegate to the 2-arg version
        results, n_samples = MacroEconometricModels.process_posterior_samples(
            post, post.p, post.n,
            (m, Q, h) -> MacroEconometricModels.compute_irf(m, Q, h);
            horizon=5, method=:cholesky
        )
        @test n_samples > 0
        @test length(results) == n_samples

        _tprint("Deprecated process_posterior_samples wrapper test passed.")
    end

    @testset "Base.size and Base.length for BVARPosterior" begin
        rng = MersenneTwister(50006)  # DGP-03: explicit rng
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 25 : 50), sampler=:direct, rng=rng)

        # length
        @test length(post) == post.n_draws

        # size(post, 1) == n_draws
        @test size(post, 1) == post.n_draws

        # size(post, 2) should error
        @test_throws ErrorException size(post, 2)

        _tprint("Base.size / Base.length tests passed.")
    end

    @testset "varnames() accessor" begin
        rng = MersenneTwister(50007)  # DGP-03: explicit rng
        # Default varnames
        post_def = estimate_bvar(Y, 1; n_draws=(FAST ? 20 : 40), sampler=:direct, rng=rng)
        vn = varnames(post_def)
        @test vn isa Vector{String}
        @test length(vn) == 2

        # Custom varnames
        post_custom = estimate_bvar(Y, 1; n_draws=(FAST ? 20 : 40), sampler=:direct, rng=rng,
                                    varnames=["X1", "X2"])
        @test varnames(post_custom) == ["X1", "X2"]

        _tprint("varnames() accessor tests passed.")
    end

    @testset "compute_posterior_quantiles with central=:median" begin
        rng = MersenneTwister(50008)  # DGP-03: explicit rng
        # Create synthetic samples array: n_samples x dim1 x dim2
        samples = randn(rng, Float64, 100, 5, 3)

        q_vec = [0.16, 0.5, 0.84]
        q_out, m_out = MacroEconometricModels.compute_posterior_quantiles(
            samples, q_vec; central=:median
        )

        # Check output shapes
        @test size(q_out) == (5, 3, 3)   # (dim1, dim2, n_quantiles)
        @test size(m_out) == (5, 3)       # (dim1, dim2)

        # m_out should be median (not mean) of each slice
        for i in 1:5, j in 1:3
            @test m_out[i, j] ≈ median(samples[:, i, j])
        end

        # Quantiles should be ordered
        for i in 1:5, j in 1:3
            @test q_out[i, j, 1] <= q_out[i, j, 2] <= q_out[i, j, 3]
        end

        # Compare with central=:mean
        q_out2, m_out2 = MacroEconometricModels.compute_posterior_quantiles(
            samples, q_vec; central=:mean
        )
        for i in 1:5, j in 1:3
            @test m_out2[i, j] ≈ mean(samples[:, i, j])
        end

        _tprint("compute_posterior_quantiles central=:median tests passed.")
    end

    @testset "Minnesota prior edge cases" begin
        rng = MersenneTwister(50009)  # DGP-03: explicit rng
        Y_mn = randn(rng, 80, 2)

        # lambda=0 and mu=0: these disable sum-of-coefficients and co-persistence priors
        hyper_no_soc = MinnesotaHyperparameters(tau=0.5, decay=2.0, lambda=0.0, mu=0.0, omega=0.5)
        post_no_soc = estimate_bvar(Y_mn, 1; prior=:minnesota, hyper=hyper_no_soc, n_draws=50, rng=rng)
        @test post_no_soc isa BVARPosterior
        @test all(isfinite.(post_no_soc.B_draws))

        # Very tight prior (small tau)
        hyper_tight = MinnesotaHyperparameters(tau=0.01, decay=2.0, omega=0.5)
        post_tight = estimate_bvar(Y_mn, 1; prior=:minnesota, hyper=hyper_tight, n_draws=50, rng=rng)
        @test post_tight isa BVARPosterior
        @test all(isfinite.(post_tight.B_draws))

        # Very loose prior (large tau)
        hyper_loose = MinnesotaHyperparameters(tau=10.0, decay=1.0, omega=1.0)
        post_loose = estimate_bvar(Y_mn, 1; prior=:minnesota, hyper=hyper_loose, n_draws=50, rng=rng)
        @test post_loose isa BVARPosterior
        @test all(isfinite.(post_loose.B_draws))

        _tprint("Minnesota prior edge cases tests passed.")
    end

    @testset "log_marginal_likelihood" begin
        rng = MersenneTwister(50010)  # DGP-03: explicit rng
        Y_lml = randn(rng, 80, 2)

        # Standard hyper
        hyper = MinnesotaHyperparameters(tau=0.5, decay=2.0, omega=0.5)
        ml = log_marginal_likelihood(Y_lml, 1, hyper)
        @test isfinite(ml)
        @test ml isa Float64

        # Different tau should give different marginal likelihoods
        hyper2 = MinnesotaHyperparameters(tau=5.0, decay=2.0, omega=0.5)
        ml2 = log_marginal_likelihood(Y_lml, 1, hyper2)
        @test isfinite(ml2)
        @test ml != ml2  # different hyperparameters should yield different values

        # optimize_hyperparameters should return valid result
        best_hyper = MacroEconometricModels.optimize_hyperparameters(Y_lml, 1; grid_size=5)
        @test best_hyper isa MinnesotaHyperparameters
        @test best_hyper.tau > 0

        # Full grid optimization
        best_full, best_ml = MacroEconometricModels.optimize_hyperparameters_full(
            Y_lml, 1;
            tau_grid=range(0.1, 2.0, length=3),
            lambda_grid=[1.0, 5.0],
            mu_grid=[1.0, 2.0]
        )
        @test best_full isa MinnesotaHyperparameters
        @test isfinite(best_ml)

        # F-02 regression: the returned value must be the TRUE Normal-Inverse-Wishart marginal
        # likelihood, including the multivariate-gamma + log-π normalization terms (previously
        # dropped). Compare against an independent assembly via the Sims `matrictint` integral:
        #   logML = matrictint(post) − matrictint(prior) − ½·T_eff·n·log(2π).
        _matrictint(S, df, XXi) = begin
            kk = size(XXi, 1); ny = size(S, 1)
            cx = cholesky(Symmetric(XXi)).U; cs = cholesky(Symmetric(S)).U
            w1 = 0.5*kk*ny*log(2π) + ny*sum(log.(diag(cx)))
            lgg = sum(loggamma(0.5*(df + 1 - j)) for j in 1:ny)
            w1 + (-df*sum(log.(diag(cs))) + 0.5*df*ny*log(2) + ny*(ny-1)*0.25*log(π) + lgg)
        end
        hyp = MinnesotaHyperparameters(tau=0.5, decay=2.0, omega=0.5)
        Yd, Xd = MacroEconometricModels.gen_dummy_obs(Y_lml, 1, hyp)
        Yeff, Xreg = MacroEconometricModels.construct_var_matrices(Y_lml, 1)
        Teff = size(Yeff, 1); kreg = size(Xreg, 2); n2 = size(Y_lml, 2)
        Yaug, Xaug = vcat(Yeff, Yd), vcat(Xreg, Xd)
        Kpost, Kprior, Td = Xaug'Xaug, Xd'Xd, size(Yd, 1)
        Baug = Kpost \ (Xaug'Yaug); Bpr = Kprior \ (Xd'Yd)
        Spost = (Yaug - Xaug*Baug)' * (Yaug - Xaug*Baug)
        Spr   = (Yd - Xd*Bpr)' * (Yd - Xd*Bpr)
        nupr = Td - kreg; nupost = Teff + nupr
        true_ml = _matrictint(Spost, nupost, inv(Kpost)) - _matrictint(Spr, nupr, inv(Kprior)) -
                  0.5*Teff*n2*log(2π)
        @test isapprox(log_marginal_likelihood(Y_lml, 1, hyp), true_ml; rtol=1e-8)

        _tprint("log_marginal_likelihood tests passed.")
    end
end

@testset "BVAR forecast companion/history reuse (#210 box B, box C)" begin
    # Box B preallocates the stability-check companion once (writing the invariant identity
    # sub-block a single time, overwriting only the top AR blocks per draw); box C replaces the
    # per-step `vcat` history ring with an in-place row shift. Both are pure-allocation refactors
    # that must leave the companion eigenvalues and the propagated history bit-for-bit identical.

    # (box B) The preallocated-and-reused companion produces eigenvalues identical to a freshly
    # built companion, for every draw, so the stationarity gate never diverges.
    n, p = 2, 3
    rng = Random.MersenneTwister(101)
    draws = [[0.2 .* randn(rng, n, n) for _ in 1:p] for _ in 1:5]
    comp_reuse = zeros(n * p, n * p)
    if p > 1
        comp_reuse[n+1:end, 1:n*(p-1)] = Matrix{Float64}(I, n*(p-1), n*(p-1))
    end
    for A_list in draws
        comp_fresh = zeros(n * p, n * p)
        for lag in 1:p
            comp_fresh[1:n, (lag-1)*n+1:lag*n] = A_list[lag]
        end
        if p > 1
            comp_fresh[n+1:end, 1:n*(p-1)] = Matrix{Float64}(I, n*(p-1), n*(p-1))
        end
        for lag in 1:p
            comp_reuse[1:n, (lag-1)*n+1:lag*n] = A_list[lag]
        end
        @test comp_reuse == comp_fresh
        @test eigvals(comp_reuse) == eigvals(comp_fresh)
    end

    # (box C) The in-place history ring shift reproduces the `vcat`-rebuilt ring exactly.
    hist_vcat = randn(rng, p, n)
    hist_inpl = copy(hist_vcat)
    for step in 1:8
        y_hat = randn(rng, n)
        # old: rebuild via vcat
        hist_vcat = vcat(@view(hist_vcat[2:end, :]), y_hat')
        # new: in-place shift
        for r in 1:(p - 1)
            @views hist_inpl[r, :] .= hist_inpl[r + 1, :]
        end
        @views hist_inpl[p, :] .= y_hat
        @test hist_inpl == hist_vcat
    end

    # Integration: the actual forecast() is deterministic on a fixed seed (the refactor did not
    # change RNG consumption), and produces finite, ordered bands.
    rngd = Random.MersenneTwister(20210)
    # Persistent VAR(2) truth (DGP-03 #792: shared simulator).
    Yd = dgp_var(rngd; A=[0.5 0.1; 0.05 0.4], B0=Matrix{Float64}(I, n, n), T=140).Y
    post = estimate_bvar(Yd, 2; n_draws=(FAST ? 40 : 80), sampler=:direct,
                         rng=MersenneTwister(9090))
    fc1 = forecast(post, 6; rng=MersenneTwister(31337))
    fc2 = forecast(post, 6; rng=MersenneTwister(31337))
    @test fc1.forecast == fc2.forecast
    @test fc1.ci_lower == fc2.ci_lower
    @test fc1.ci_upper == fc2.ci_upper
    @test all(isfinite, fc1.forecast)
    @test all(fc1.ci_upper .>= fc1.ci_lower)
end

@testset "BVAR IRF MC honesty counts (#244)" begin
    rng = MersenneTwister(4244)  # DGP-03: explicit rng
    Y = randn(rng, 120, 2)
    post = estimate_bvar(Y, 2; n_draws=80, sampler=:direct, rng=rng)
    b = irf(post, 8; method=:cholesky)
    @test b.n_requested == 80
    @test b.n_effective + b.n_failed == b.n_requested
    @test 0 <= b.n_failed <= b.n_requested

    # cumulation is a deterministic transform of the same draws — counts propagate
    bc = cumulative_irf(b)
    @test (bc.n_requested, bc.n_effective, bc.n_failed) == (b.n_requested, b.n_effective, b.n_failed)

    # backward-compatible constructors: 6-arg ⇒ zero counts; 7-arg infers from draw count
    q = zeros(8, 2, 2, 3); pe = zeros(8, 2, 2); vars = ["y1", "y2"]; shk = ["s1", "s2"]; ql = [0.16, 0.5, 0.84]
    b6 = MacroEconometricModels.BayesianImpulseResponse{Float64}(q, pe, 8, vars, shk, ql)
    @test (b6.n_requested, b6.n_effective, b6.n_failed) == (0, 0, 0)
    b7 = MacroEconometricModels.BayesianImpulseResponse{Float64}(q, pe, 8, vars, shk, ql, zeros(50, 8, 2, 2))
    @test (b7.n_requested, b7.n_effective, b7.n_failed) == (50, 50, 0)

    # display surfaces dropped draws only when some were dropped
    b_drop = MacroEconometricModels.BayesianImpulseResponse{Float64}(q, pe, 8, vars, shk, ql, nothing, 100, 60, 40)
    s = sprint(show, b_drop)
    @test occursin("Effective draws", s) && occursin("60/100", s) && occursin("40 dropped", s)
    @test !occursin("Effective draws", sprint(show, b))

    # --- BayesianFEVD counts ---
    f = fevd(post, 8; method=:cholesky)
    @test f.n_requested == 80
    @test f.n_effective + f.n_failed == f.n_requested
    # Synthetic arrays use unified (variable, shock, horizon) order (#527)
    q_fevd = zeros(2, 2, 8, 3); pe_fevd = zeros(2, 2, 8)
    f6 = MacroEconometricModels.BayesianFEVD{Float64}(q_fevd, pe_fevd, 8, vars, shk, ql)
    @test (f6.n_requested, f6.n_effective, f6.n_failed) == (0, 0, 0)
    f_drop = MacroEconometricModels.BayesianFEVD{Float64}(q_fevd, pe_fevd, 8, vars, shk, ql, 100, 70, 30)
    sf = sprint(show, f_drop)
    @test occursin("70/100", sf) && occursin("30 dropped", sf)
    @test !occursin("Effective draws", sprint(show, f))

    # --- BayesianHistoricalDecomposition counts ---
    hd = historical_decomposition(post; method=:cholesky)
    @test hd.n_requested == 80
    @test hd.n_effective + hd.n_failed == hd.n_requested
    Te = hd.T_eff; q4 = zeros(Te, 2, 2, 3); p3 = zeros(Te, 2, 2); q3 = zeros(Te, 2, 3); m2 = zeros(Te, 2)
    h11 = MacroEconometricModels.BayesianHistoricalDecomposition{Float64}(
        q4, p3, q3, m2, m2, m2, Te, vars, shk, ql, :cholesky)
    @test (h11.n_requested, h11.n_effective, h11.n_failed) == (0, 0, 0)
    hd_drop = MacroEconometricModels.BayesianHistoricalDecomposition{Float64}(
        q4, p3, q3, m2, m2, m2, Te, vars, shk, ql, :cholesky, 100, 70, 30)
    sh = sprint(show, hd_drop)
    @test occursin("70/100", sh) && occursin("30 dropped", sh)
    @test !occursin("Effective draws", sprint(show, hd))
end

@testset "SID-07 IdentificationError in posterior loops" begin
    rng = MersenneTwister(736)  # DGP-03: explicit rng
    Y = randn(rng, 80, 2)
    post = estimate_bvar(Y, 1; n_draws=FAST ? 20 : 40, burnin=10, rng=rng)
    impossible(irf) = irf[1, 1, 1] > 1e6
    @test_throws IdentificationError irf(post, 5; method=:sign, check_func=impossible, max_draws=3)
    @test_throws IdentificationError historical_decomposition(post; method=:sign, check_func=impossible, max_draws=3)
    pos(irf) = irf[1, 1, 1] > 0
    r = irf(post, 5; method=:sign, check_func=pos, max_draws=20)
    @test r isa BayesianImpulseResponse
    # n_failed = unidentified + non-stationary (SID-07). Both counts appear in the warning
    # and, after SID-19, on BayesianSetIdentifiedSVAR.n_unidentified.
    @test r.n_failed == r.n_requested - r.n_effective
    # Minority identification failures must skip the draw, not abort the posterior loop.
    # n_failed > 0 is the SID-07 contract; n_failed == n_requested - n_effective is constructor arithmetic.
    r1 = irf(post, 5; method=:sign, check_func=pos, max_draws=1)
    @test r1 isa BayesianImpulseResponse
    @test r1.n_failed == r1.n_requested - r1.n_effective
    @test r1.n_failed > 0
    @test r1.n_effective > 0
    hd1 = historical_decomposition(post; method=:sign, check_func=pos, max_draws=1)
    @test hd1 isa BayesianHistoricalDecomposition
    @test hd1.n_failed == hd1.n_requested - hd1.n_effective
    @test hd1.n_failed > 0
    @test hd1.n_effective > 0

    # identify_arias_bayesian: skip unidentified posterior draws; throw only if all fail.
    restr = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive)])
    ar1 = identify_arias_bayesian(post, restr, 5; n_rotations=1)
    @test ar1 isa BayesianSetIdentifiedSVAR
    @test ar1.n_unidentified > 0
    @test ar1.total_accepted > 0
    @test ar1.n_degenerate_weights >= 0
    restr_imp = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive),
                                           sign_restriction(1, 1, :negative)])
    @test_throws IdentificationError identify_arias_bayesian(post, restr_imp, 5; n_rotations=3)

    # SID-19: irf/fevd(post; method=:arias) is identify_arias_bayesian, not compute_Q.
    rng_a = MersenneTwister(748)
    ir_arias = irf(post, 5; method=:arias, restrictions=restr, max_draws=1, rng=copy(rng_a))
    ar_ref = identify_arias_bayesian(post, restr, 5; n_rotations=1, rng=copy(rng_a))
    @test ir_arias isa BayesianImpulseResponse
    @test ir_arias.quantiles ≈ irf(ar_ref).quantiles
    rng_f = MersenneTwister(749)
    fv_arias = fevd(post, 5; method=:arias, restrictions=restr, max_draws=1, rng=copy(rng_f))
    ar_f = identify_arias_bayesian(post, restr, 5; n_rotations=1, rng=copy(rng_f))
    @test fv_arias isa BayesianFEVD
    @test fv_arias.quantiles ≈ fevd(ar_f).quantiles
    err = try
        historical_decomposition(post; method=:arias, restrictions=restr)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("identify_arias_bayesian", err.msg)
end

@testset "SID-04 fallback P_ref when posterior-mean Q unidentified" begin
    # Noiseless VAR so the posterior-mean B fits U ≈ 0. Smooth-transition ID
    # calls `_eigendecomposition_id` on the split covariances (no λ-gap
    # fallback), so identical ~eps I regimes throw IdentificationError on the
    # mean; perturbed draws remain identified and become P_ref.
    # (`:external_volatility` swallows the λ-gap error in `_two_regime_start`.)
    n, p, Tobs = 2, 1, 80
    A = [0.5 0.0; 0.0 0.5]
    Y = zeros(Tobs, n)
    Y[1, :] = [2.0, -1.5]
    for t in 2:Tobs
        Y[t, :] = A * Y[t-1, :]
    end
    B_true = [0.0 0.0; 0.5 0.0; 0.0 0.5]
    Δ = [0.0 0.0; 0.0 0.15; 0.15 0.0]
    B_draws = zeros(3, 3, 2)
    B_draws[1, :, :] = B_true
    B_draws[2, :, :] = B_true + Δ
    B_draws[3, :, :] = B_true - Δ
    Sigma_draws = zeros(3, 2, 2)
    for s in 1:3
        Sigma_draws[s, :, :] .= Matrix(1.0I, 2, 2)
    end
    post = BVARPosterior{Float64}(B_draws, Sigma_draws, 3, p, n, Y, :normal, :direct, ["y1", "y2"])
    s = collect(range(-2.0, 2.0; length=Tobs))
    results, ns = @test_logs (:warn, r"reference Q could not be identified") match_mode = :any begin
        MacroEconometricModels.process_posterior_samples(post,
            (m, Q, h) -> MacroEconometricModels.compute_irf(m, Q, h);
            method=:smooth_transition, horizon=4, transition_var=s)
    end
    @test ns >= 2
    @test length(results) == ns
    @test all(r -> size(r) == (4, n, n) && all(isfinite, r), results)
end

@testset "SID-18 identify_robust_bayes on BVARPosterior" begin
    rng = MersenneTwister(747)  # DGP-03: explicit rng
    Y = randn(rng, 50, 2)
    post = estimate_bvar(Y, 1; n_draws=FAST ? 6 : 10, burnin=3, seed=747)
    r = SVARRestrictions(2; signs=[sign_restriction(1, 1, :positive),
                                   sign_restriction(2, 1, :positive)])
    nrot = FAST ? 8 : 16
    rng = MersenneTwister(747)
    res = identify_robust_bayes(post, r, 2; level=0.68, solver=:optimize,
                                n_rotations=nrot, rng=copy(rng))
    @test res isa RobustBayesResult
    @test res.empty_set_prob == 0
    @test 0 <= res.informativeness <= 1
    # Single-prior interval is identify_arias_bayesian (all nonempty draws,
    # pooled weights, equal-tailed). GK guarantees Haar coverage of the CR,
    # not that the equal-tailed Haar interval ⊂ CR.
    rand(rng, UInt64, post.n_draws)
    q_lo = (1 - 0.68) / 2
    arias = identify_arias_bayesian(post, r, 2; n_rotations=nrot,
                                    quantiles=[q_lo, 0.5, 1 - q_lo],
                                    compute_weights=true, rng=rng)
    @test res.single_prior_lower ≈ arias.irf_quantiles[:, :, :, 1]
    @test res.single_prior_upper ≈ arias.irf_quantiles[:, :, :, end]
    w = arias.weights
    n_acc = size(arias.irf_draws, 1)
    H, n, _ = size(res.robust_lower)
    for h in 1:H, i in 1:n, j in 1:n
        cl = res.robust_lower[h, i, j]
        cu = res.robust_upper[h, i, j]
        (isfinite(cl) && isfinite(cu)) || continue
        mass = zero(eltype(w))
        for s in 1:n_acc
            η = arias.irf_draws[s, h, i, j]
            if cl - 1e-10 <= η <= cu + 1e-10
                mass += w[s]
            end
        end
        @test mass >= res.level - 1e-12
    end
end
