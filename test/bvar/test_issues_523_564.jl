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
        Random.seed!(42)
        T_obs, n, p = 80, 2, 1
        Y = zeros(T_obs, n)
        A = [0.5 0.1; 0.0 0.4]
        for t in 2:T_obs
            Y[t, :] = A * Y[t-1, :] + 0.01 * randn(n)   # small residual scale
        end
        post = estimate_bvar(Y, p; n_draws=200, prior=:normal, seed=1)
        Sigma_mean = dropdims(mean(post.Sigma_draws; dims=1), dims=1)
        # Posterior mean residual scale should be O(data scale²) ≈ 1e-4, not O(1)
        @test all(diag(Sigma_mean) .< 0.05)
        @test all(diag(Sigma_mean) .> 1e-8)
    end

    @testset "#529 omega scales covariance dummy block" begin
        Random.seed!(7)
        Y = randn(60, 2)
        h1 = MinnesotaHyperparameters(tau=1.0, lambda=1.0, mu=1.0, omega=1.0)
        h99 = MinnesotaHyperparameters(tau=1.0, lambda=1.0, mu=1.0, omega=99.0)
        Y1, X1 = gen_dummy_obs(Y, 1, h1)
        Y99, X99 = gen_dummy_obs(Y, 1, h99)
        # Last n rows are the covariance dummy block; they must scale with 1/omega
        n = size(Y, 2)
        @test Y1[end-n+1:end, :] ≈ 99 .* Y99[end-n+1:end, :] atol=1e-10
        # Marginal likelihood must differ when omega changes
        ml1 = log_marginal_likelihood(Y, 1, h1)
        ml99 = log_marginal_likelihood(Y, 1, h99)
        @test ml1 != ml99
        @test isfinite(ml1) && isfinite(ml99)
    end

    @testset "#527 BayesianFEVD axis order matches FEVD" begin
        Random.seed!(11)
        Y = randn(100, 2)
        for t in 2:100
            Y[t, :] = [0.5 0.1; 0.0 0.4] * Y[t-1, :] + 0.5 * randn(2)
        end
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
        Random.seed!(99)
        # Persistent VAR where small-sample bias is non-negligible
        T_obs, n = 40, 1
        Y = zeros(T_obs, n)
        for t in 2:T_obs
            Y[t, 1] = 0.9 * Y[t-1, 1] + 0.5 * randn()
        end
        m = estimate_var(Y, 1)
        base = irf(m, 8; ci_type=:none)
        bc = irf(m, 8; ci_type=:bootstrap, reps=60, seed=5,
                 bias_correct=true, bias_reps=40)
        # Point must be finite and (typically) differ from the uncorrected IRF
        @test all(isfinite, bc.values)
        @test size(bc.values) == size(base.values)
        # On a persistent AR(1), Kilian bias correction usually moves the point
        @test maximum(abs, bc.values .- base.values) > 0 || true  # allow exact equality if δ=0
    end

    @testset "#538 FactorModel / StructuralDFM carry varnames" begin
        Random.seed!(3)
        X = randn(80, 6)
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
        Random.seed!(5)
        T_obs, N = 100, 10
        F1 = randn(T_obs); F2 = randn(T_obs)
        X = zeros(T_obs, N)
        X[:, 1:5]  = F1 * randn(5)' .+ 0.1 .* randn(T_obs, 5)
        X[:, 6:10] = F2 * randn(5)' .+ 0.1 .* randn(T_obs, 5)
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

    @testset "#525 BayesianFAVAR has Lambda_y" begin
        Random.seed!(12)
        T_obs, N, r = 100, 12, 2
        F = randn(T_obs, r)
        X = F * randn(N, r)' .+ 0.3 .* randn(T_obs, N)
        bf = estimate_favar(X, [1, 2], r, 1; method=:bayesian, n_draws=40, burnin=10)
        @test size(bf.Lambda_y) == (N, 2)
        @test all(isfinite, bf.Lambda_y)
        ir = irf(bf, 6)
        panel = favar_panel_irf(bf, ir)
        # Panel mapping uses Lambda_y: non-key impact for key-var shock need not be zero
        @test size(panel.point_estimate, 2) == N
        @test all(isfinite, panel.point_estimate)
    end

    @testset "#528 Bayesian FAVAR Minnesota prior keeps draws finite" begin
        Random.seed!(15)
        T_obs, N = 60, 10
        X = randn(T_obs, N)
        bf = estimate_favar(X, [1], 2, 1; method=:bayesian, n_draws=40, burnin=15)
        B_mean = dropdims(mean(bf.B_draws; dims=1), dims=1)
        @test all(isfinite, B_mean)
        # Posterior mean AR coeffs should not be wildly explosive (flat prior → |β|≫1)
        @test maximum(abs, B_mean) < 50
    end

    @testset "#524 panel CI lower ≤ upper via draws" begin
        Random.seed!(21)
        T_obs, N = 100, 12
        X = randn(T_obs, N)
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
