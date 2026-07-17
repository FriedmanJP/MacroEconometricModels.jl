# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# Tests for Markov-switching regression / mean-switching AR (EV-07 / #415):
# the scaled Hamilton (1989) forward filter, the Kim (1994) smoother, EM, and the
# Optim ML polish with delta-method SEs.
#
# Oracle discipline
# -----------------
# (1) ANALYTIC PROPERTIES (primary): transition rows sum to 1; smoothed and
#     filtered probabilities lie in [0,1] and sum to 1 across regimes at every t;
#     the ergodic initial vector equals the stationary distribution of P
#     (π'P = π'); the logit↔P round-trip and expanded-state helpers are exact;
#     regime labelling (order by increasing mean) is deterministic across RNG
#     seeds.
# (2) KNOWN-VALUE oracle: Hamilton (1989, Table I) MS(2)-AR(4) US GNP estimates
#         μ_recession ≈ -0.358, μ_expansion ≈ 1.164, σ ≈ 0.769,
#         p_expansion-stay ≈ 0.904, q_recession-stay ≈ 0.755,
#         φ ≈ (0.014, -0.058, -0.247, -0.213)
#     asserted with loose tolerance on load_example(:gnp_hamilton).
# (3) CROSS-IMPLEMENTATION oracle (statsmodels 0.14.4, computed externally on the
#     SAME rgnp series; statsmodels IS installed at ~/miniconda3 but the numerics
#     are pinned here as documented constants per test/oracle/ convention):
#       sm.tsa.MarkovRegression(rgnp, k_regimes=2, trend="c", switching_variance=True).fit()
#         const  = [-0.22427, 1.17650]      (statsmodels labels regime 0 = low mean)
#         sigma2 = [ 0.94235, 0.61975]
#         P[0,0] =  0.75307,  P[1,1] = 0.89212
#         llf    = -190.68737
#     Exact call recorded for reproduction:
#       from statsmodels.tsa.regime_switching.tests import test_markov_autoregression as tm
#       import statsmodels.api as sm; rgnp = np.array(tm.rgnp)
#       sm.tsa.MarkovRegression(rgnp, k_regimes=2, trend="c", switching_variance=True).fit()

using Test
using MacroEconometricModels
using Random
using Statistics
using LinearAlgebra
using Distributions

const MEM = MacroEconometricModels

# -----------------------------------------------------------------------------
# Synthetic mean-switching MS-AR(1) generator (fixed seed, explicit rng)
# -----------------------------------------------------------------------------
# (yₜ − μ_{sₜ}) = φ (y_{t−1} − μ_{s_{t−1}}) + εₜ,  εₜ ~ N(0, σ²).
function _sim_ms_ar1(rng; n=600, mu=(-1.0, 3.0), phi=0.4, sigma=0.6,
                     P=[0.9 0.1; 0.15 0.85])
    s = Vector{Int}(undef, n)
    s[1] = 1
    for t in 2:n
        s[t] = rand(rng) < P[s[t-1], 1] ? 1 : 2
    end
    z = zeros(n)          # deviation from regime mean
    y = zeros(n)
    for t in 2:n
        z[t] = phi * z[t-1] + sigma * randn(rng)
        y[t] = mu[s[t]] + z[t]
    end
    return y[2:end], s[2:end]
end

@testset "EV-07 Markov-switching regression / MS-AR" begin

    # =========================================================================
    # Internal helpers — exact analytic identities
    # =========================================================================
    @testset "helper identities" begin
        # logit ↔ P round-trip (row-softmax, last column reference).
        for K in (2, 3)
            Random.seed!(11)
            rows = [rand(Dirichlet(ones(K))) for _ in 1:K]
            P = permutedims(hcat(rows...))                 # K×K, rows sum to 1
            l = MEM._P_to_logits(P, K)
            P2 = MEM._logits_to_P(l, K)
            @test isapprox(P, P2; atol=1e-10)
            @test all(isapprox.(sum(P2, dims=2), 1.0; atol=1e-12))
        end

        # Ergodic vector: π'P = π', Σπ = 1, π ≥ 0.
        P = [0.9 0.1; 0.2 0.8]
        pi_ = MEM._ms_ergodic(P)
        @test isapprox(sum(pi_), 1.0; atol=1e-12)
        @test all(pi_ .>= -1e-12)
        @test isapprox(P' * pi_, pi_; atol=1e-10)          # stationary
        # Closed form for a 2-state chain: π = (1-p22, 1-p11)/(2-p11-p22).
        @test isapprox(pi_[1], (1 - 0.8) / (2 - 0.9 - 0.8); atol=1e-10)

        # Expanded state space K^(p+1): shape, current-regime marginal, transitions.
        K, p = 2, 2
        states, allowed = MEM._expand_states(K, p)
        @test size(states) == (K^(p+1), p+1)
        @test sort(unique(states)) == collect(1:K)
        Pk = [0.7 0.3; 0.25 0.75]
        Pexp = MEM._expanded_P(Pk, states, BitMatrix(allowed))
        @test all(isapprox.(sum(Pexp, dims=2), 1.0; atol=1e-12))  # still stochastic
        # Marginalisation sums a distribution back to 1.
        M = size(states, 1)
        prob = rand(5, M); prob ./= sum(prob, dims=2)
        marg = MEM._ms_marginalise(prob, states, K)
        @test all(isapprox.(sum(marg, dims=2), 1.0; atol=1e-12))
        @test size(marg) == (5, K)
    end

    # =========================================================================
    # Filter / smoother analytic properties
    # =========================================================================
    @testset "filter & smoother properties" begin
        y = vec(load_example(:gnp_hamilton).data)
        m = estimate_ms(y; k_regimes=2, switching_variance=true)

        # Probabilities are valid distributions at every t.
        for probs in (m.filtered_prob, m.smoothed_prob)
            @test all(0 .- 1e-9 .<= probs .<= 1 .+ 1e-9)
            @test all(isapprox.(sum(probs, dims=2), 1.0; atol=1e-8))
        end
        # Transition rows sum to 1.
        @test all(isapprox.(sum(m.P, dims=2), 1.0; atol=1e-10))
        # Ergodic field equals the stationary vector of the fitted P.
        @test isapprox(m.ergodic, MEM._ms_ergodic(m.P); atol=1e-10)
        @test isapprox(m.P' * m.ergodic, m.ergodic; atol=1e-9)
        # Expected durations 1/(1-p_kk).
        @test isapprox(m.expected_durations[1], 1 / (1 - m.P[1, 1]); atol=1e-10)
    end

    # =========================================================================
    # estimate_ms — cross-implementation oracle (statsmodels MarkovRegression)
    # =========================================================================
    @testset "estimate_ms vs statsmodels (GNP, switching mean+var)" begin
        y = vec(load_example(:gnp_hamilton).data)
        m = estimate_ms(y; k_regimes=2, switching_variance=true)
        @test m.model_type == :regression
        @test m.k_regimes == 2
        # Regimes ordered by increasing mean → regime 1 low, regime 2 high.
        @test m.mu[1] < m.mu[2]
        # statsmodels: const = [-0.22427, 1.17650].
        @test isapprox(m.mu[1], -0.22427; atol=0.02)
        @test isapprox(m.mu[2],  1.17650; atol=0.02)
        # statsmodels: sigma2 = [0.94235, 0.61975].
        @test isapprox(m.sigma2[1], 0.94235; atol=0.03)
        @test isapprox(m.sigma2[2], 0.61975; atol=0.03)
        # statsmodels: P[0,0]=0.75307, P[1,1]=0.89212.
        @test isapprox(m.P[1, 1], 0.75307; atol=0.02)
        @test isapprox(m.P[2, 2], 0.89212; atol=0.02)
        # statsmodels: llf = -190.68737.
        @test isapprox(m.loglik, -190.68737; atol=0.2)
    end

    # =========================================================================
    # estimate_ms_ar — Hamilton (1989, Table I) known-value oracle
    # =========================================================================
    @testset "estimate_ms_ar vs Hamilton (1989) GNP replication" begin
        y = vec(load_example(:gnp_hamilton).data)
        m = estimate_ms_ar(y, 4; k_regimes=2)
        @test m.model_type == :ms_ar
        @test m.p == 4
        @test m.n == length(y) - 4
        # Regimes ordered by mean: regime 1 = recession, regime 2 = expansion.
        @test m.mu[1] < m.mu[2]
        # Hamilton Table I: μ_recession ≈ -0.358, μ_expansion ≈ 1.164 (loose bands).
        @test isapprox(m.mu[1], -0.358; atol=0.15)
        @test isapprox(m.mu[2],  1.164; atol=0.15)
        # p_expansion-stay ≈ 0.904, q_recession-stay ≈ 0.755.
        @test isapprox(m.P[2, 2], 0.904; atol=0.06)
        @test isapprox(m.P[1, 1], 0.755; atol=0.08)
        # σ ≈ 0.769  ⇒  σ² ≈ 0.591.
        @test isapprox(sqrt(m.sigma2[1]), 0.769; atol=0.06)
        # φ ≈ (0.014, -0.058, -0.247, -0.213).
        @test isapprox(m.ar, [0.014, -0.058, -0.247, -0.213]; atol=0.05)
        # Reported SEs are finite and positive.
        @test all(isfinite, m.se_coefs) && all(m.se_coefs .>= 0)
        @test all(isfinite, m.se_ar)   && all(m.se_ar .>= 0)
        @test m.converged
    end

    # =========================================================================
    # Determinism: label ordering stable across RNG seeds; estimation is seed-free
    # =========================================================================
    @testset "deterministic labelling / seed independence" begin
        y1, _ = _sim_ms_ar1(MersenneTwister(101))
        y2, _ = _sim_ms_ar1(MersenneTwister(202))
        m1 = estimate_ms_ar(y1, 1; k_regimes=2)
        m2 = estimate_ms_ar(y2, 1; k_regimes=2)
        # Low-mean regime is ALWAYS regime 1 regardless of the data seed.
        @test m1.mu[1] < m1.mu[2]
        @test m2.mu[1] < m2.mu[2]
        # Estimation itself carries no RNG: re-fitting identical data is bit-identical.
        m1b = estimate_ms_ar(y1, 1; k_regimes=2)
        @test m1.mu == m1b.mu
        @test m1.P == m1b.P
        @test m1.loglik == m1b.loglik
    end

    # =========================================================================
    # Recovery on a synthetic mean-switching DGP with known parameters
    # =========================================================================
    @testset "parameter recovery (synthetic MS-AR1)" begin
        y, strue = _sim_ms_ar1(MersenneTwister(7); n=800, mu=(-1.0, 3.0),
                               phi=0.4, sigma=0.6, P=[0.92 0.08; 0.12 0.88])
        m = estimate_ms_ar(y, 1; k_regimes=2)
        @test isapprox(m.mu[1], -1.0; atol=0.4)
        @test isapprox(m.mu[2],  3.0; atol=0.4)
        @test isapprox(m.ar[1],  0.4; atol=0.15)
        @test isapprox(sqrt(m.sigma2[1]), 0.6; atol=0.15)
        @test isapprox(m.P[1, 1], 0.92; atol=0.08)
        @test isapprox(m.P[2, 2], 0.88; atol=0.08)
        # Smoothed regime classification recovers the true path reasonably well.
        pred = [m.smoothed_prob[t, 2] > 0.5 ? 2 : 1 for t in 1:m.n]
        strue_eff = strue[2:end]                        # AR(1) drops one obs
        acc = mean(pred .== strue_eff)
        @test acc > 0.85
    end

    # =========================================================================
    # estimate_ms with an explicit design (switching slope) + switching_variance flag
    # =========================================================================
    @testset "estimate_ms with regressors and non-switching variance" begin
        rng = MersenneTwister(33)
        n = 500
        x = randn(rng, n)
        s = Vector{Int}(undef, n); s[1] = 1
        P = [0.95 0.05; 0.07 0.93]
        for t in 2:n
            s[t] = rand(rng) < P[s[t-1], 1] ? 1 : 2
        end
        b1 = [0.0, 1.0]; b2 = [4.0, -0.5]
        y = [ (s[t]==1 ? b1[1]+b1[2]*x[t] : b2[1]+b2[2]*x[t]) + 0.5*randn(rng) for t in 1:n ]
        X = hcat(ones(n), x)
        m = estimate_ms(y, X; k_regimes=2, switching_variance=false,
                        xnames=["const", "x"])
        # Non-switching variance ⇒ equal σ² entries.
        @test isapprox(m.sigma2[1], m.sigma2[2]; atol=1e-6)
        @test !m.switching_var
        @test size(m.coefs) == (2, 2)
        # Regime intercepts recovered (regime 1 low mean ⇒ intercept ≈ 0).
        @test isapprox(m.coefs[1, 1], 0.0; atol=0.6)
        @test isapprox(m.coefs[1, 2], 4.0; atol=0.6)
        @test m.xnames == ["const", "x"]
    end

    # =========================================================================
    # StatsAPI accessors, display, refs, plotting smoke tests
    # =========================================================================
    @testset "accessors / display / refs / plot" begin
        y = vec(load_example(:gnp_hamilton).data)
        m = estimate_ms_ar(y, 2; k_regimes=2)

        @test nobs(m) == m.n
        @test length(residuals(m)) == m.n
        @test loglikelihood(m) == m.loglik
        @test dof(m) == m.n_params
        @test aic(m) > 0 && bic(m) > 0
        @test length(coef(m)) == m.k_regimes + m.p

        # report() runs and emits a nonempty table.
        io = IOBuffer(); report(io, m); out = String(take!(io))
        @test occursin("Markov-Switching AR", out)
        @test occursin("Transition matrix", out)

        # refs() lists Hamilton (1989) and Kim (1994).
        rio = IOBuffer(); refs(rio, m); rout = String(take!(rio))
        @test occursin("Hamilton", rout)
        @test occursin("Kim", rout)

        # plot_result dispatches for both views.
        p1 = plot_result(m; view=:probabilities)
        p2 = plot_result(m; view=:filtered)
        @test p1 isa MEM.PlotOutput
        @test p2 isa MEM.PlotOutput
        @test_throws ArgumentError plot_result(m; view=:nope)
    end

    # =========================================================================
    # Argument validation
    # =========================================================================
    @testset "argument validation" begin
        y = vec(load_example(:gnp_hamilton).data)
        @test_throws ArgumentError estimate_ms_ar(y, 0)          # p ≥ 1
        @test_throws ArgumentError estimate_ms_ar(y, 4; k_regimes=1)
        @test_throws ArgumentError estimate_ms(y; k_regimes=1)
        @test_throws ArgumentError estimate_ms_ar(randn(6), 4)   # too short
        @test_throws DimensionMismatch estimate_ms(y, ones(length(y) + 3, 1))
    end

    # =========================================================================
    # Dataset loader
    # =========================================================================
    @testset "load_example(:gnp_hamilton)" begin
        ts = load_example(:gnp_hamilton)
        @test size(ts.data, 1) == 135
        @test ts.varnames == ["gnp_growth"]
        @test isapprox(vec(ts.data)[1], 2.5932; atol=1e-4)
        @test isapprox(vec(ts.data)[end], 0.1480; atol=1e-4)
    end
end
