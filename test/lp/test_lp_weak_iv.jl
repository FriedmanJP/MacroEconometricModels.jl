# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# T245 (#344): weak-IV-robust LP-IV inference — Montiel Olea-Pflueger effective F and
# horizon-wise Anderson-Rubin bands.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics
using Distributions

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _M = MacroEconometricModels
const _suppress_warnings = MacroEconometricModels._suppress_warnings

"""LP-IV DGP: `pi1` sets instrument strength, `theta` the impact response of y to the shock."""
function _lpiv_sim(T_obs::Int; pi1::Float64=1.5, seed::Int=1, theta::Float64=1.0)
    rng = Random.MersenneTwister(seed)
    z = randn(rng, T_obs)
    v = randn(rng, T_obs)
    s = pi1 .* z .+ v                     # shock, endogenous w.r.t. the y equation
    y = zeros(T_obs)
    x2 = zeros(T_obs)
    for t in 2:T_obs
        y[t] = 0.5 * y[t-1] + theta * s[t] + 0.6 * v[t] + randn(rng)
        x2[t] = 0.3 * x2[t-1] + 0.4 * s[t] + randn(rng)
    end
    return hcat(s, y, x2), reshape(z, :, 1)
end

@testset "LP-IV weak-instrument-robust inference" begin

# ─────────────────────────────────────────────────────────────────────────────
# Montiel Olea-Pflueger effective F
# ─────────────────────────────────────────────────────────────────────────────

@testset "effective F: algebraic reduction and hand computation" begin
    Y, Z = _lpiv_sim(400; pi1=1.5, seed=3)
    m = estimate_lp_iv(Y, 1, Z, 4; lags=2, varnames=["s", "y", "x2"])

    mop = montiel_olea_pflueger_f(m)
    @test mop isa MontielOleaPfluegerF{Float64}
    @test mop.n_instruments == 1
    @test mop.tau == 0.10
    @test mop.critical_value == 23.11
    @test isfinite(mop.f_effective) && mop.f_effective > 0
    @test !mop.weak                              # a strong instrument by construction

    # Independent hand computation of the numerator and denominator from the residualized
    # first stage, using the same Newey-West sandwich.
    _, endog, Zh, W = _M._lp_iv_horizon_pieces(m, 0)
    Zt = _M._partial_out(Zh, W)
    xt = _M._partial_out(endog, W)
    pi_hat = (Zt' * Zt) \ (Zt' * xt)
    resid = xt .- Zt * pi_hat
    bw = mop.bandwidth
    V = _M.newey_west(Matrix(Zt), resid; bandwidth=bw)
    signal = dot(xt, Zt * pi_hat)
    noise = tr(V * (Zt' * Zt))
    @test mop.f_effective ≈ signal / noise atol = 1e-10

    # THE defining property (Montiel Olea-Pflueger): substituting the HOMOSKEDASTIC
    # covariance σ̂²(Z̃'Z̃)⁻¹ makes the denominator exactly q·σ̂², so the effective F
    # collapses to the classical first-stage F.
    n = length(xt)
    q = size(Zh, 2)
    kW = size(W, 2)
    sigma2 = dot(resid, resid) / (n - q - kW)
    V_homo = sigma2 * inv(Matrix(Zt' * Zt))
    @test tr(V_homo * (Zt' * Zt)) ≈ q * sigma2 atol = 1e-10
    classical_f = signal / (q * sigma2)
    @test signal / tr(V_homo * (Zt' * Zt)) ≈ classical_f atol = 1e-10

    # With a data-driven HAC bandwidth on near-homoskedastic data the two stay close
    @test mop.f_effective ≈ classical_f rtol = 0.3
end

@testset "effective F flags a weak instrument" begin
    Y, Z = _lpiv_sim(200; pi1=0.05, seed=7)
    m = estimate_lp_iv(Y, 1, Z, 3; lags=2, varnames=["s", "y", "x2"])
    mop = montiel_olea_pflueger_f(m)
    @test mop.weak
    @test mop.f_effective < mop.critical_value

    out = sprint(show, mop)
    @test occursin("Montiel Olea-Pflueger", out)
    @test occursin("WEAK instruments", out)
    @test occursin("lp_iv_ar_band", out)
    @test report(mop) === nothing
end

@testset "effective F: bias targets and validation" begin
    Y, Z = _lpiv_sim(300; pi1=1.5, seed=3)
    m = estimate_lp_iv(Y, 1, Z, 2; lags=2)

    # The tabulated MOP simplified critical values are decreasing in the bias tolerance
    cvs = [montiel_olea_pflueger_f(m; tau=t).critical_value for t in (0.05, 0.10, 0.20, 0.30)]
    @test issorted(cvs; rev=true)
    @test cvs == [37.42, 23.11, 15.06, 12.04]
    # The statistic itself does not depend on the target
    fs = [montiel_olea_pflueger_f(m; tau=t).f_effective for t in (0.05, 0.10, 0.20, 0.30)]
    @test all(f -> f ≈ fs[1], fs)

    # A fixed bandwidth is honored
    @test montiel_olea_pflueger_f(m; bandwidth=6).bandwidth == 6

    @test_throws ArgumentError montiel_olea_pflueger_f(m; tau=0.15)
end

# ─────────────────────────────────────────────────────────────────────────────
# Horizon-wise Anderson-Rubin bands
# ─────────────────────────────────────────────────────────────────────────────

@testset "strong instrument: AR band tracks the Wald band" begin
    Y, Z = _lpiv_sim(400; pi1=1.5, seed=3, theta=1.0)
    m = estimate_lp_iv(Y, 1, Z, 5; lags=2, varnames=["s", "y", "x2"])
    band = lp_iv_ar_band(m; responses=[2], n_grid=201)

    @test band isa LPIVARBand{Float64}
    @test band.horizon == 5
    @test size(band.lower) == (6, 1)
    @test band.response_names == ["y"]
    @test all(band.bounded)
    @test !any(band.is_empty)

    # Point estimates agree with lp_iv_irf, and the AR band brackets them
    wald = lp_iv_irf(m; conf_level=0.95)
    @test band.point[:, 1] ≈ wald.values[:, 2] atol = 1e-12
    @test all(band.lower[:, 1] .< band.point[:, 1] .< band.upper[:, 1])

    # With a strong first stage the AR and Wald bands are close at every horizon
    @test maximum(abs, band.lower[:, 1] .- band.wald_lower[:, 1]) < 0.05
    @test maximum(abs, band.upper[:, 1] .- band.wald_upper[:, 1]) < 0.05

    # The impact response recovers the true theta = 1.0
    @test band.lower[1, 1] < 1.0 < band.upper[1, 1]
end

@testset "HAC lag length scales with the horizon" begin
    Y, Z = _lpiv_sim(300; pi1=1.5, seed=11)
    m = estimate_lp_iv(Y, 1, Z, 12; lags=2)
    band = lp_iv_ar_band(m; responses=[2], n_grid=51)

    @test size(band.bandwidths) == (13, 1)
    # The horizon-h LP residual is MA(h) by construction, so the lag length is at least
    # h+1 at every horizon — the same rule estimate_lp_iv applies to its own SEs.
    for h in 0:12
        @test band.bandwidths[h+1, 1] >= h + 1
    end
    # ... and the binding h+1 floor makes the sequence strictly grow at long horizons
    @test band.bandwidths[end, 1] > band.bandwidths[1, 1]

    # A fixed bandwidth overrides the rule
    band_fixed = lp_iv_ar_band(m; responses=[2], n_grid=21, bandwidth=4)
    @test all(==(4), band_fixed.bandwidths)
end

@testset "weak instrument: AR band is wider than Wald and unbounded" begin
    Y, Z = _lpiv_sim(200; pi1=0.05, seed=7)
    m = estimate_lp_iv(Y, 1, Z, 4; lags=2, varnames=["s", "y", "x2"])
    @test montiel_olea_pflueger_f(m).weak

    band = lp_iv_ar_band(m; responses=[2], n_grid=201)
    @test !all(band.bounded)                     # at least one cell is unbounded
    # Every Wald cell is finite, so the AR band is strictly less informative — correctly so
    @test all(isfinite, band.wald_lower)
    @test all(isfinite, band.wald_upper)
    for h in 1:size(band.lower, 1)
        if !band.bounded[h, 1]
            ar_w = band.upper[h, 1] - band.lower[h, 1]
            wald_w = band.wald_upper[h, 1] - band.wald_lower[h, 1]
            @test ar_w > wald_w
        end
    end

    out = sprint(show, band)
    @test occursin("Anderson-Rubin", out)
    @test occursin("∞", out)
    @test occursin("over-confident", out)
    @test report(band) === nothing
end

@testset "coverage under a weak instrument" begin
    nrep = FAST ? 30 : 100
    theta_true = 1.0
    cov_ar = 0
    cov_wald = 0
    for s in 1:nrep
        Y, Z = _lpiv_sim(150; pi1=0.1, seed=3000 + s, theta=theta_true)
        m = _suppress_warnings() do
            estimate_lp_iv(Y, 1, Z, 1; lags=2)
        end
        band = _suppress_warnings() do
            lp_iv_ar_band(m; responses=[2], n_grid=101, span=60)
        end
        # Impact response only — the horizon at which theta_true is exactly identified
        cov_ar += any(a <= theta_true <= b for (a, b) in band.sets[1, 1])
        cov_wald += band.wald_lower[1, 1] <= theta_true <= band.wald_upper[1, 1]
    end
    # AR keeps nominal coverage where the Wald band does not
    @test cov_ar / nrep >= 0.90
    @test cov_ar >= cov_wald
end

@testset "multiple responses and validation" begin
    Y, Z = _lpiv_sim(300; pi1=1.5, seed=3)
    m = estimate_lp_iv(Y, 1, Z, 3; lags=2, varnames=["s", "y", "x2"])

    band_all = lp_iv_ar_band(m; n_grid=51)
    @test size(band_all.lower) == (4, 3)
    @test length(band_all.response_names) == 3
    # Selecting a subset reproduces the corresponding column
    band_one = lp_iv_ar_band(m; responses=[2], n_grid=51)
    @test band_one.point[:, 1] ≈ band_all.point[:, 2] atol = 1e-12
    @test band_one.lower[:, 1] ≈ band_all.lower[:, 2] atol = 1e-8
    @test band_one.bandwidths[:, 1] == band_all.bandwidths[:, 2]

    # A wider level gives a weakly wider band
    b95 = lp_iv_ar_band(m; responses=[2], n_grid=101)
    b99 = lp_iv_ar_band(m; responses=[2], n_grid=101, level=0.99)
    @test b99.critical_value > b95.critical_value
    @test all(b99.upper[:, 1] .>= b95.upper[:, 1] .- 1e-8)
    @test all(b99.lower[:, 1] .<= b95.lower[:, 1] .+ 1e-8)

    @test_throws ArgumentError lp_iv_ar_band(m; level=1.5)
    @test_throws ArgumentError lp_iv_ar_band(m; n_grid=2)
    @test_throws ArgumentError lp_iv_ar_band(m; responses=[9])
end

@testset "the T013 Sargan-J fix is untouched" begin
    # Two instruments so the model is over-identified and the J statistic is defined.
    Y, Z1 = _lpiv_sim(300; pi1=1.2, seed=17)
    rng = Random.MersenneTwister(99)
    Z2 = 0.9 .* Z1 .+ 0.3 .* randn(rng, size(Z1, 1), 1)
    m = estimate_lp_iv(Y, 1, hcat(Z1, Z2), 2; lags=2)
    sj = sargan_test(m, 0)
    @test sj.valid
    @test sj.df == 1
    @test isfinite(sj.J_stat) && sj.J_stat >= 0
    # A valid over-identifying restriction is not rejected
    @test sj.p_value > 0.01
end

@testset "Sargan-J is correctly SIZED at h=0 (degenerate own-shock equation)" begin
    # A single seed cannot catch a mis-sized test: the original one-draw assertion
    # above passed on ~1 seed in 12 while the h=0 statistic rejected valid
    # instruments almost always. The cause is that at h=0 the shock variable's
    # response to ITSELF is a regression of a series on itself — it fits exactly,
    # σ̂² is ~1e-31, and J = (...)/σ̂² exploded into the cross-equation average.
    # Assert the SIZE of the test over many draws, which is the property that
    # actually matters and does not depend on any one RNG stream.
    for h in 0:2
        n_rej = 0
        nrep = 30
        for seed in 1:nrep
            Y, Z1 = _lpiv_sim(300; pi1=1.2, seed=seed)
            rng = Random.MersenneTwister(1000 + seed)
            Z2 = 0.9 .* Z1 .+ 0.3 .* randn(rng, size(Z1, 1), 1)
            m = estimate_lp_iv(Y, 1, hcat(Z1, Z2), 3; lags=2)
            sj = sargan_test(m, h)
            @test sj.valid && isfinite(sj.J_stat) && sj.J_stat >= 0
            n_rej += sj.p_value < 0.01
        end
        # Nominal size is 1%; before the fix h=0 rejected ~92% of draws.
        @test n_rej <= 3
    end

    # The degenerate equation is dropped, not silently zeroed: the h=0 residual
    # variance of the shock's own response is numerically zero by construction.
    Y, Z1 = _lpiv_sim(300; pi1=1.2, seed=5)
    rng = Random.MersenneTwister(11)
    Z2 = 0.9 .* Z1 .+ 0.3 .* randn(rng, size(Z1, 1), 1)
    m = estimate_lp_iv(Y, 1, hcat(Z1, Z2), 2; lags=2)
    U0 = m.residuals[1]
    s2 = [sum(abs2, @view U0[:, e]) / m.T_eff[1] for e in 1:size(U0, 2)]
    @test s2[m.shock_var] < 1e-20            # the shock on itself
    @test all(s2[e] > 1e-3 for e in 1:length(s2) if e != m.shock_var)
end

end  # @testset "LP-IV weak-instrument-robust inference"
