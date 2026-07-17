# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# Tests for the public state-space module (EV-37, #445).
#
# Oracle layers:
#   (1) PUBLISHED known-value — Durbin & Koopman (2012), Time Series Analysis by State
#       Space Methods, 2nd ed., §2: the Nile local-level MLE recovers
#       σ̂²_ε ≈ 15099 and σ̂²_η ≈ 1469.1. Asserted at ~2% relative tolerance.
#   (2) LIVE cross-implementation — statsmodels UnobservedComponents(nile,'local level')
#       run in this environment (miniconda python3, statsmodels installed). Exact call:
#         mod = sm.tsa.UnobservedComponents(nile, 'local level'); res = mod.fit(disp=False)
#       gave params (sigma2.irregular, sigma2.level) = (15078.01, 1478.81),
#       filtered_state[0:3] = [1103.363, 1132.813, 1067.949],
#       filtered_state[-1] = 798.085. Both statsmodels and this module use large-κ
#       approximate-diffuse initialization (κ≈1e6), so the filtered path matches;
#       the log-likelihood is NOT compared (statsmodels drops the diffuse obs from
#       the likelihood, whereas the κI prior keeps a -½log κ offset here).
#   (3) ANALYTIC — P_smooth ≤ P_filt elementwise on the diagonal; standardized
#       one-step residuals pass a Ljung–Box check on well-specified data;
#       estimate_tvp_reg recovers a known time-varying slope on a fixed-seed
#       random-walk-coefficient DGP.

using Test, MacroEconometricModels, Random, LinearAlgebra, Statistics, Distributions

@testset "State-Space Module (EV-37)" begin

    # ------------------------------------------------------------------
    # Nile local level — published + live cross-impl oracles
    # ------------------------------------------------------------------
    @testset "Nile local level (Durbin–Koopman 2012 + statsmodels)" begin
        nile = load_example(:nile)
        m = local_level(nile)

        @test m isa StateSpaceModel
        @test MacroEconometricModels.isfitted(m)
        @test m.method === :mle
        @test m.T_obs == 100
        @test m.n_state == 1
        @test length(m.theta) == 2

        σ2_ε, σ2_η = m.theta
        # (1) Published Durbin–Koopman (2012, §2) MLE variances, ~2% rel-tol.
        @test isapprox(σ2_ε, 15099.0; rtol=0.02)
        @test isapprox(σ2_η, 1469.1;  rtol=0.02)

        # (2) Live statsmodels filtered-state path (a_{t|t}), κ≈1e6 approx-diffuse.
        @test isapprox(m.filtered_state[1, 1], 1103.363; rtol=1e-3)
        @test isapprox(m.filtered_state[2, 1], 1132.813; rtol=1e-3)
        @test isapprox(m.filtered_state[3, 1], 1067.949; rtol=2e-3)
        @test isapprox(m.filtered_state[end, 1], 798.085; rtol=1e-3)

        # (3) Analytic: smoothed variance ≤ filtered variance on the diagonal.
        @test all(m.smoothed_cov[1, 1, t] <= m.filtered_cov[1, 1, t] + 1e-6
                  for t in 1:m.T_obs)

        # Standardized one-step residuals ~ white noise (well-specified model).
        r = vec(m.std_residuals); r = r[.!isnan.(r)]
        lb = ljung_box_test(r; lags=10)
        @test lb.pvalue > 0.05
        # Standardized residuals have ~unit variance (drop the diffuse first obs).
        @test isapprox(std(r[2:end]), 1.0; atol=0.15)
    end

    # ------------------------------------------------------------------
    # Vector input + local_linear_trend
    # ------------------------------------------------------------------
    @testset "local_linear_trend + vector input" begin
        nile = load_example(:nile)
        y = vec(nile.data)
        m = local_linear_trend(y)
        @test m isa StateSpaceModel
        @test m.n_state == 2
        @test length(m.theta) == 3
        @test all(m.theta .>= 0)                     # variances non-negative
        @test all(m.smoothed_cov[1, 1, t] <= m.filtered_cov[1, 1, t] + 1e-6
                  for t in 1:m.T_obs)
        @test isfinite(m.loglik)
    end

    # ------------------------------------------------------------------
    # TVP regression — recover a known random-walk slope (analytic oracle)
    # ------------------------------------------------------------------
    @testset "estimate_tvp_reg random-walk-slope recovery" begin
        rng = MersenneTwister(20240717)
        T = 300
        x = randn(rng, T)
        beta = zeros(T); beta[1] = 1.0
        for t in 2:T
            beta[t] = beta[t-1] + 0.03 * randn(rng)
        end
        alpha = 0.5
        y = alpha .+ beta .* x .+ 0.2 .* randn(rng, T)

        m = estimate_tvp_reg(y, reshape(x, :, 1))
        @test m isa StateSpaceModel
        @test m.n_state == 2                          # intercept + slope
        @test size(m.smoothed_state) == (T, 2)

        bhat = m.smoothed_state[:, 2]                 # slope path (col 2)
        ahat = m.smoothed_state[:, 1]                 # intercept path (col 1)
        @test cor(bhat, beta) > 0.9                   # recovers the time-varying slope
        @test sqrt(mean((bhat .- beta) .^ 2)) < 0.15
        @test isapprox(mean(ahat), alpha; atol=0.1)   # fixed intercept recovered

        # σ̂²_ε near the true irregular variance (0.2² = 0.04).
        @test isapprox(m.theta[1], 0.04; rtol=0.5)

        # Well-specified: standardized residuals pass Ljung–Box.
        r = vec(m.std_residuals); r = r[.!isnan.(r)]
        @test ljung_box_test(r; lags=10).pvalue > 0.05
    end

    # ------------------------------------------------------------------
    # Fixed-matrix constructor + filter path == parametric fit at same θ
    # ------------------------------------------------------------------
    @testset "fixed-matrix spec + estimate_statespace(ss, y)" begin
        nile = load_example(:nile)
        y = vec(nile.data)
        ll = local_level(y)                            # get MLE variances

        spec = StateSpaceModel(reshape([1.0], 1, 1), reshape([ll.theta[1]], 1, 1),
                               reshape([1.0], 1, 1), reshape([ll.theta[2]], 1, 1))
        @test spec isa StateSpaceModel
        @test spec.method === :spec
        @test !MacroEconometricModels.isfitted(spec)

        f = estimate_statespace(spec, y)
        @test f.method === :filter
        @test MacroEconometricModels.isfitted(f)
        # Same system + data ⇒ identical log-likelihood and filtered states as the fit.
        @test isapprox(f.loglik, ll.loglik; rtol=1e-10)
        @test isapprox(f.filtered_state, ll.filtered_state; rtol=1e-8)
    end

    # ------------------------------------------------------------------
    # Generic parametric estimate_statespace with a user builder (AR(1) obs)
    # ------------------------------------------------------------------
    @testset "generic estimate_statespace user builder" begin
        # Stationary AR(1) in the state, observed with noise:
        #   αₜ = φ αₜ₋₁ + ηₜ,  yₜ = αₜ + εₜ.  θ = [φ, log σ²_η, log σ²_ε].
        rng = MersenneTwister(7)
        T = 400
        φ = 0.7; ση = 1.0; σε = 0.5
        α = zeros(T)
        for t in 2:T
            α[t] = φ * α[t-1] + ση * randn(rng)
        end
        y = α .+ σε .* randn(rng, T)
        # tanh maps ℝ → (-1,1) smoothly, keeping the AR(1) state stationary with a
        # well-behaved gradient (a raw clamp would flatten the objective at ±0.99).
        build = θ -> (Z=reshape([1.0], 1, 1), H=reshape([exp(θ[3])], 1, 1),
                      T=reshape([tanh(θ[1])], 1, 1),
                      Q=reshape([exp(θ[2])], 1, 1))
        m = estimate_statespace(build, [0.3, 0.0, 0.0], y;
                                param_names=["atanh(φ)", "logσ²_η", "logσ²_ε"])
        @test m isa StateSpaceModel
        @test isapprox(tanh(m.theta[1]), φ; atol=0.15)   # φ recovered
        @test isfinite(m.loglik)
    end

    # ------------------------------------------------------------------
    # Forecast — state recursion + variance accumulation
    # ------------------------------------------------------------------
    @testset "forecast" begin
        nile = load_example(:nile)
        m = local_level(nile)
        h = 6
        fc = forecast(m, h)
        @test size(fc.mean) == (h, 1)
        @test size(fc.se) == (h, 1)
        # Local level: flat point forecast at the last filtered level.
        @test all(isapprox.(fc.mean[:, 1], m.filtered_state[end, 1]; rtol=1e-6))
        # Predictive SE strictly increasing with horizon (variance accumulates).
        @test issorted(fc.se[:, 1])
        @test fc.se[end, 1] > fc.se[1, 1]
        @test_throws ArgumentError forecast(m, 0)
    end

    # ------------------------------------------------------------------
    # Display: report / refs / show run without error
    # ------------------------------------------------------------------
    @testset "report / refs / show" begin
        nile = load_example(:nile)
        m = local_level(nile)
        io = IOBuffer()
        report(io, m)
        s = String(take!(io))
        @test occursin("State-Space Model", s)
        @test occursin("σ²_ε", s)
        @test occursin("Log-likelihood", s)

        io2 = IOBuffer(); refs(io2, m)
        @test occursin("Durbin", String(take!(io2)))

        io3 = IOBuffer(); show(io3, m)
        @test occursin("StateSpaceModel", String(take!(io3)))

        # Unfitted spec renders too.
        spec = StateSpaceModel(reshape([1.0], 1, 1), reshape([1.0], 1, 1),
                               reshape([1.0], 1, 1), reshape([1.0], 1, 1))
        io4 = IOBuffer(); report(io4, spec)
        @test occursin("unfitted", String(take!(io4)))

        # plot_result renders a PlotOutput.
        p = plot_result(m)
        @test p isa MacroEconometricModels.PlotOutput
        @test_throws ArgumentError plot_result(spec)   # unfitted → error
    end

    # ------------------------------------------------------------------
    # Constructor validation
    # ------------------------------------------------------------------
    @testset "constructor argument checks" begin
        Z = reshape([1.0], 1, 1); H = reshape([1.0], 1, 1)
        Tt = reshape([1.0], 1, 1); Q = reshape([1.0], 1, 1)
        @test_throws ArgumentError StateSpaceModel(Z, reshape([1.0,1.0], 2, 1), Tt, Q)  # bad H
        @test_throws ArgumentError StateSpaceModel(Z, H, Tt, Q; d=[0.0, 0.0])           # bad d length
        # Builder missing a required field.
        @test_throws ArgumentError StateSpaceModel(θ -> (Z=Z, H=H, T=Tt), [0.0])
    end
end
