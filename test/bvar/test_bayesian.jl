# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
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

using MacroEconometricModels
using Test
using LinearAlgebra
using Statistics
using Random

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

@testset "BVAR Bayesian Parameter Recovery" begin
    _tprint("Generating Data for Bayesian Verification...")

    # 1. Generate Synthetic Data
    T = 100
    n = 2
    p = 1
    Random.seed!(42)

    true_A = [0.5 0.0; 0.0 0.5]
    true_c = [0.0; 0.0]

    Y = zeros(T, n)
    for t in 2:T
        u = randn(2)  # Unit variance
        Y[t, :] = true_c + true_A * Y[t-1, :] + u
    end

    # 2. Direct Sampler Parameter Recovery (Primary Test)
    @testset "Direct Sampler Parameter Recovery" begin
        _tprint("Estimating BVAR (direct)...")
        post = estimate_bvar(Y, p; n_draws=(FAST ? 30 : 100), sampler=:direct)
        @test post isa BVARPosterior

        # Extract and check parameter recovery
        b_vecs, _ = MacroEconometricModels.extract_chain_parameters(post)
        means_arr = vec(mean(b_vecs, dims=1))

        _tprint("Recovered Means: ", means_arr)

        # Check intercepts (should be near 0)
        @test abs(means_arr[1]) < 0.5
        @test abs(means_arr[4]) < 0.5

        # Check diagonal A elements (should be near 0.5)
        @test isapprox(means_arr[2], 0.5, atol=0.35)
        @test isapprox(means_arr[6], 0.5, atol=0.35)

        # Check off-diagonal A elements (should be near 0)
        @test abs(means_arr[3]) < 0.35
        @test abs(means_arr[5]) < 0.35

        _tprint("Direct Sampler Parameter Recovery Verified.")
    end

    # 3. Gibbs Sampler Smoke Test
    @testset "Gibbs Sampler Smoke Test" begin
        _tprint("Estimating BVAR (Gibbs)...")
        post_gibbs = estimate_bvar(Y, p;
            n_draws=(FAST ? 20 : 50), sampler=:gibbs, burnin=(FAST ? 20 : 50), thin=1
        )
        @test post_gibbs isa BVARPosterior
        @test post_gibbs.n_draws == (FAST ? 20 : 50)
        @test post_gibbs.sampler == :gibbs
        _tprint("Gibbs Sampler Smoke Test Passed.")
    end

    # ==========================================================================
    # Robustness Tests
    # ==========================================================================

    @testset "Reproducibility" begin
        _tprint("Testing BVAR reproducibility...")
        Random.seed!(77777)
        Y_rep = zeros(80, 2)
        for t in 2:80
            Y_rep[t, :] = 0.5 * Y_rep[t-1, :] + randn(2)
        end

        Random.seed!(88888)
        post1 = estimate_bvar(Y_rep, 1; n_draws=50, sampler=:direct)

        Random.seed!(88888)
        post2 = estimate_bvar(Y_rep, 1; n_draws=50, sampler=:direct)

        # Same random seed should give same results
        @test post1.B_draws ≈ post2.B_draws
        @test post1.Sigma_draws ≈ post2.Sigma_draws
        _tprint("Reproducibility test passed.")
    end

    @testset "Numerical Stability - Near-Collinear Data" begin
        _tprint("Testing numerical stability with near-collinear data...")
        Random.seed!(11111)
        T_nc = 80
        n_nc = 3

        # Create data with near-collinearity
        Y_nc = randn(T_nc, n_nc)
        Y_nc[:, 3] = Y_nc[:, 1] + 0.01 * randn(T_nc)

        post_nc = estimate_bvar(Y_nc, 1; n_draws=50, sampler=:direct)
        @test post_nc isa BVARPosterior

        # Check all parameters are finite
        @test all(isfinite.(post_nc.B_draws))
        _tprint("Numerical stability test passed.")
    end

    @testset "Edge Cases" begin
        _tprint("Testing edge cases...")
        Random.seed!(22222)

        # Single variable BVAR
        Y_single = randn(80, 1)
        post_single = estimate_bvar(Y_single, 1; n_draws=50)
        @test post_single isa BVARPosterior

        # Verify parameter dimensions for single variable
        # k = 1 + n*p = 1 + 1*1 = 2
        @test size(post_single.B_draws, 2) == 2  # intercept + 1 AR coefficient
        @test size(post_single.B_draws, 3) == 1  # 1 variable
        _tprint("Edge case tests passed.")
    end

    @testset "Posterior Draws Structure" begin
        _tprint("Testing posterior draws structure...")
        Random.seed!(33333)
        Y_diag = zeros(80, 2)
        for t in 2:80
            Y_diag[t, :] = 0.5 * Y_diag[t-1, :] + randn(2)
        end

        post_diag = estimate_bvar(Y_diag, 1; n_draws=50, sampler=:direct)

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
        Random.seed!(44444)
        Y_post = zeros(80, 2)
        for t in 2:80
            Y_post[t, :] = 0.5 * Y_post[t-1, :] + randn(2)
        end

        post = estimate_bvar(Y_post, 1; n_draws=50)

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
        Random.seed!(99887)
        Y_mn = randn(80, 2)
        hyper = MinnesotaHyperparameters(tau=0.2, decay=2.0, omega=0.5)
        post_mn = estimate_bvar(Y_mn, 1; prior=:minnesota, hyper=hyper, n_draws=100)
        @test post_mn isa BVARPosterior
        @test post_mn.prior == :minnesota
        _tprint("Minnesota prior BVAR test passed.")
    end

    @testset "BVAR sampler variants" begin
        Random.seed!(99886)
        Y_sv = randn(60, 2)

        # Direct sampler
        @testset "Direct sampler" begin
            post_direct = estimate_bvar(Y_sv, 1; sampler=:direct, n_draws=50)
            @test post_direct isa BVARPosterior
            @test post_direct.sampler == :direct
            _tprint("Direct sampler test passed.")
        end

        # Gibbs sampler
        @testset "Gibbs sampler" begin
            post_gibbs = estimate_bvar(Y_sv, 1; sampler=:gibbs, n_draws=50, burnin=100)
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
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 30 : 50), sampler=:direct)
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
        Random.seed!(50001)
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 30 : 80), sampler=:direct)

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
        @test fc.point_estimate == :median  # default
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
        Random.seed!(50002)
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 30 : 60), sampler=:direct)

        # Show with :median (default)
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
        Random.seed!(50003)
        post_vn = estimate_bvar(Y, 1; n_draws=(FAST ? 30 : 50), sampler=:direct,
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
        Random.seed!(50004)
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 30 : 50), sampler=:direct)

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
        Random.seed!(50005)
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 20 : 40), sampler=:direct)

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
        Random.seed!(50006)
        post = estimate_bvar(Y, 1; n_draws=(FAST ? 25 : 50), sampler=:direct)

        # length
        @test length(post) == post.n_draws

        # size(post, 1) == n_draws
        @test size(post, 1) == post.n_draws

        # size(post, 2) should error
        @test_throws ErrorException size(post, 2)

        _tprint("Base.size / Base.length tests passed.")
    end

    @testset "varnames() accessor" begin
        Random.seed!(50007)
        # Default varnames
        post_def = estimate_bvar(Y, 1; n_draws=(FAST ? 20 : 40), sampler=:direct)
        vn = varnames(post_def)
        @test vn isa Vector{String}
        @test length(vn) == 2

        # Custom varnames
        post_custom = estimate_bvar(Y, 1; n_draws=(FAST ? 20 : 40), sampler=:direct,
                                    varnames=["X1", "X2"])
        @test varnames(post_custom) == ["X1", "X2"]

        _tprint("varnames() accessor tests passed.")
    end

    @testset "compute_posterior_quantiles with central=:median" begin
        Random.seed!(50008)
        # Create synthetic samples array: n_samples x dim1 x dim2
        samples = randn(Float64, 100, 5, 3)

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
        Random.seed!(50009)
        Y_mn = randn(80, 2)

        # lambda=0 and mu=0: these disable sum-of-coefficients and co-persistence priors
        hyper_no_soc = MinnesotaHyperparameters(tau=0.5, decay=2.0, lambda=0.0, mu=0.0, omega=0.5)
        post_no_soc = estimate_bvar(Y_mn, 1; prior=:minnesota, hyper=hyper_no_soc, n_draws=50)
        @test post_no_soc isa BVARPosterior
        @test all(isfinite.(post_no_soc.B_draws))

        # Very tight prior (small tau)
        hyper_tight = MinnesotaHyperparameters(tau=0.01, decay=2.0, omega=0.5)
        post_tight = estimate_bvar(Y_mn, 1; prior=:minnesota, hyper=hyper_tight, n_draws=50)
        @test post_tight isa BVARPosterior
        @test all(isfinite.(post_tight.B_draws))

        # Very loose prior (large tau)
        hyper_loose = MinnesotaHyperparameters(tau=10.0, decay=1.0, omega=1.0)
        post_loose = estimate_bvar(Y_mn, 1; prior=:minnesota, hyper=hyper_loose, n_draws=50)
        @test post_loose isa BVARPosterior
        @test all(isfinite.(post_loose.B_draws))

        _tprint("Minnesota prior edge cases tests passed.")
    end

    @testset "log_marginal_likelihood" begin
        Random.seed!(50010)
        Y_lml = randn(80, 2)

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

        _tprint("log_marginal_likelihood tests passed.")
    end
end
