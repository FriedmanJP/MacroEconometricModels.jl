# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# Regression tests for issues #523, #524, #525, #526, #527, #528, #529, #538, #563, #564.

using Test
using LinearAlgebra
using Statistics
using Random
using MacroEconometricModels

const M = MacroEconometricModels

@testset "Issue fixes #523–#564" begin

    @testset "#523 NIW S0 is scale-invariant" begin
        rng = MersenneTwister(42)  # DGP-03: explicit rng
        p = 1
        # Small residual scale truth (DGP-03 #792: shared simulator).
        Y = dgp_var(rng; A=[0.5 0.1; 0.0 0.4], Sigma=Matrix(1e-4 * I, 2, 2), T=80).Y
        post = estimate_bvar(Y, p; n_draws=200, prior=:normal, seed=1)
        Sigma_mean = dropdims(mean(post.Sigma_draws; dims=1), dims=1)
        # Posterior mean residual scale should be O(data scale²) ≈ 1e-4, not O(1)
        @test all(diag(Sigma_mean) .< 0.05)
        @test all(diag(Sigma_mean) .> 1e-8)
    end

    @testset "#529 omega replicates the covariance dummy block" begin
        rng = MersenneTwister(7)  # DGP-03: explicit rng
        # Stationary VAR(1) truth (DGP-03 #792) — the dummy algebra only needs
        # a well-conditioned Y for the residual scale.
        Y = dgp_var(rng; A=[0.5 0.1; 0.0 0.4], B0=Matrix{Float64}(I, 2, 2), T=60).Y
        n = size(Y, 2)
        h0 = MinnesotaHyperparameters(tau=1.0, lambda=1.0, mu=1.0, omega=0.0)
        h1 = MinnesotaHyperparameters(tau=1.0, lambda=1.0, mu=1.0, omega=1.0)
        h3 = MinnesotaHyperparameters(tau=1.0, lambda=1.0, mu=1.0, omega=3.0)
        Y0, _ = gen_dummy_obs(Y, 1, h0)
        Y1, _ = gen_dummy_obs(Y, 1, h1)
        Y3, _ = gen_dummy_obs(Y, 1, h3)
        # omega is a replication count: each copy adds n rows, all equal diag(σ̂)
        # (weight = prior dof around the SAME location — not a rescaled location).
        @test size(Y1, 1) == size(Y0, 1) + n
        @test size(Y3, 1) == size(Y0, 1) + 3n
        @test Y3[end-n+1:end, :] ≈ Y1[end-n+1:end, :] atol=1e-12
        @test Y3[end-2n+1:end-n, :] ≈ Y1[end-n+1:end, :] atol=1e-12
        # Marginal likelihood must respond to omega
        ml1 = log_marginal_likelihood(Y, 1, h1)
        ml3 = log_marginal_likelihood(Y, 1, h3)
        @test ml1 != ml3
        @test isfinite(ml1) && isfinite(ml3)
    end

    @testset "#527 BayesianFEVD axis order matches FEVD" begin
        rng = MersenneTwister(11)  # DGP-03: explicit rng
        # Same design as the old inline sim, on the shared simulator (DGP-03 #792).
        Y = dgp_var(rng; A=[0.5 0.1; 0.0 0.4], Sigma=Matrix(0.25 * I, 2, 2), T=100).Y
        m = estimate_var(Y, 1)
        post = estimate_bvar(Y, 1; n_draws=80, seed=3)
        f = fevd(m, 6)
        bf = fevd(post, 6)
        @test size(f.proportions) == (2, 2, 6)          # (variable, shock, horizon)
        @test size(bf.point_estimate) == (2, 2, 6)      # same order
        @test size(bf.quantiles) == (2, 2, 6, 3)
        # Proportions sum to ~1 across shocks at each (var, horizon)
        for h in 1:6, v in 1:2
            @test sum(bf.point_estimate[v, :, h]) ≈ 1.0 atol=0.15
        end
    end

    @testset "#563 report uses point_estimate not middle quantile" begin
        H, n, nq = 4, 2, 3
        # Make mean far from median quantile so the bug would be visible
        pe = fill(9.0, H, n, n)
        q = zeros(H, n, n, nq)
        q[:, :, :, 1] .= -1.0
        q[:, :, :, 2] .= 0.0     # middle quantile (what old report printed)
        q[:, :, :, 3] .= 1.0
        birf = BayesianImpulseResponse{Float64}(q, pe, H, ["a", "b"], ["s1", "s2"],
                                                [0.16, 0.5, 0.84])
        s = sprint(show, birf)
        @test occursin("9", s)          # point estimate value appears
        @test occursin("Point", s)      # label honours point estimate
    end

    @testset "#564 bias_correct corrects the point IRF" begin
        rng = MersenneTwister(99)  # DGP-03: explicit rng
        # Persistent AR(1) where small-sample bias is non-negligible
        # (DGP-03 #792: shared univariate simulator, truth φ = 0.9).
        Y = reshape(dgp_arima(rng; phi=[0.9], sigma=0.5, T=40).y, :, 1)
        m = estimate_var(Y, 1)
        base = irf(m, 8; ci_type=:none)
        bc = irf(m, 8; ci_type=:bootstrap, reps=60, seed=5,
                 bias_correct=true, bias_reps=40)
        @test all(isfinite, bc.values)
        @test size(bc.values) == size(base.values)
        # On a persistent AR(1) at T=40 the OLS bias is material — the corrected
        # point MUST differ from the uncorrected one (that was the whole bug).
        @test maximum(abs, bc.values .- base.values) > 1e-8
        # Downward OLS bias ⇒ corrected own-response at long horizon is larger
        @test bc.values[8, 1, 1] > base.values[8, 1, 1]
        # bias_correct without bootstrap machinery is an explicit error, not a
        # silently uncorrected point (#564)
        @test_throws ArgumentError irf(m, 8; ci_type=:none, bias_correct=true)
    end

    @testset "#538 FactorModel / StructuralDFM carry varnames" begin
        rng = MersenneTwister(3)  # DGP-03: explicit rng
        # Genuine 2-factor panel (DGP-03 #792) instead of white noise.
        X = dgp_dynamic_factors(rng; N=6, T=80).X
        names = ["A", "B", "C", "D", "E", "F"]
        fm = estimate_factors(X, 2; varnames=names)
        @test fm.varnames == names
        s = sprint(show, fm)
        @test occursin("A", s) || occursin("Var", s)

        sdfm = estimate_structural_dfm(X, 2; p=1, H=10, varnames=names)
        @test sdfm.varnames == names
        ir = irf(sdfm, 10)
        @test ir.variables == names
    end

    @testset "#526 block-restricted variance is per-block" begin
        rng = MersenneTwister(5)  # DGP-03: explicit rng
        # Block-restricted 2-factor truth (DGP-03 #792): factor 1 loads on
        # series 1:5, factor 2 on series 6:10.
        N = 10
        d = dgp_dynamic_factors(rng; N=N, T=100,
                                blocks=Dict(1 => 1:5, 2 => 6:10))
        X = d.X
        blocks = Dict(:real => collect(1:5), :nominal => collect(6:10))
        fm = estimate_factors(X, 2; blocks=blocks)
        @test fm.block_names !== nothing
        @test length(fm.eigenvalues) == N          # full panel spectrum retained
        block_expl = M._block_explained_variance(fm)
        # Per-block shares should be high (factors fit their blocks well)
        @test all(block_expl .> 0.5)
        @test all(block_expl .<= 1.0 + 1e-8)
        s = sprint(show, fm)
        @test occursin("per-block", s)
    end

    @testset "#525 Bayesian panel mapping is Λ·factor_irf (no Λ_y channel)" begin
        rng = MersenneTwister(12)  # DGP-03: explicit rng
        N, r = 12, 2
        # Genuine dynamic-factor panel (DGP-03 #792) instead of white noise.
        X = dgp_dynamic_factors(rng; N=N, T=100).X
        bf = estimate_favar(X, [1, 2], r, 1; method=:bayesian, n_draws=40, burnin=10)
        # The mapping is Λ · factor_irf with no Λ_y channel, because `BayesianFAVAR`
        # stores no Λ_y: the Gibbs sampler now carries one internally (#528) but treats
        # it as a nuisance parameter. NOTE: that makes the omission substantive rather
        # than a double-counting guard as originally argued here — the factors are no
        # longer free to absorb the Y_key-transmitted component, so panel responses to a
        # Y_key shock understate the direct channel. Storing Λ_y and adding it to this
        # mapping is tracked as follow-up work.
        @test !hasproperty(bf, :Lambda_y)
        ir = irf(bf, 6)
        panel = favar_panel_irf(bf, ir)
        @test size(panel.point_estimate, 2) == N
        @test all(isfinite, panel.point_estimate)
        # Non-key rows equal Λ · factor_irf exactly
        Lam = dropdims(mean(bf.loadings_draws; dims=1), dims=1)
        i_nonkey = findfirst(i -> !(i in bf.Y_key_indices), 1:N)
        for h in 1:6, j in 1:(r + bf.n_key)
            expected = dot(Lam[i_nonkey, :], ir.point_estimate[h, 1:r, j])
            @test panel.point_estimate[h, i_nonkey, j] ≈ expected atol=1e-10
        end
    end

    @testset "#528 Bayesian FAVAR Gibbs draws are bounded (flat NIW block)" begin
        # The explosive-draw symptom was a factor identification problem, not a prior
        # choice: a Minnesota dummy block recomputed from the current draw each sweep
        # is a state-dependent prior (no fixed invariant distribution) and was reverted.
        # The fix is BBE's measurement equation X = ΛF + Λ_y Y + e with Λ[anchor,:] = I,
        # which stops F from tracking Y_key and keeps the VAR design well conditioned,
        # so a magnitude bound can now be asserted alongside finiteness.
        rng = MersenneTwister(15)  # DGP-03: explicit rng
        # Genuine dynamic-factor panel (DGP-03 #792) instead of white noise.
        X = dgp_dynamic_factors(rng; N=10, T=60).X
        bf = estimate_favar(X, [1], 2, 1; method=:bayesian, n_draws=40, burnin=15)
        B_mean = dropdims(mean(bf.B_draws; dims=1), dims=1)
        @test all(isfinite, B_mean)
        @test all(isfinite, bf.Sigma_draws)
        @test maximum(abs, B_mean) < 3.0
    end

    @testset "#524 panel CI lower ≤ upper via draws" begin
        rng = MersenneTwister(21)  # DGP-03: explicit rng
        # Genuine dynamic-factor panel (DGP-03 #792) instead of white noise.
        X = dgp_dynamic_factors(rng; N=12, T=100).X
        favar = estimate_favar(X, [1, 3], 2, 1)
        ir = irf(favar, 6; ci_type=:bootstrap, reps=25, seed=9)
        panel = favar_panel_irf(favar, ir)
        @test all(panel.ci_lower .<= panel.ci_upper)
        @test panel._draws !== nothing
        fc = forecast(favar, 4; ci_method=:bootstrap, reps=25)
        pfc = favar_panel_forecast(favar, fc)
        @test all(pfc.ci_lower .<= pfc.ci_upper)
    end
end
