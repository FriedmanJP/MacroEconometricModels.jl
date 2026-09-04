# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# T251 (#350): Schorfheide-Song mixed-frequency Bayesian VAR.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _M = MacroEconometricModels

# DGP-03 (#792): latent high-frequency VAR(1) path on the shared simulator
# (same design as the GLP file's historical DGP).
const _MF_A = [0.6 0.1; 0.15 0.5]
const _MF_B0 = [0.4 0.0; 0.1 0.3]

"""Latent high-frequency VAR(1) path with known truth."""
function _mf_sim(rng::AbstractRNG, T_hf::Int)
    return dgp_var(rng; A=_MF_A, B0=_MF_B0, T=T_hf).Y
end

"""Aggregate column `col` of `Z` to the low frequency, blanking the other rows."""
function _mf_blank(Z::Matrix{Float64}, col::Int, kind::Symbol, m::Int)
    w = _M._mf_agg_weights(kind, m, Float64)
    T_hf = size(Z, 1)
    data = copy(Z)
    for t in 1:T_hf
        if t % m == 0 && t >= length(w)
            data[t, col] = sum(w[j] * Z[t-j+1, col] for j in eachindex(w))
        else
            data[t, col] = NaN
        end
    end
    return data
end

@testset "Mixed-Frequency VAR" begin

# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

@testset "temporal aggregation weights" begin
    @test _M._mf_agg_weights(:stock, 3, Float64) == [1.0]
    @test _M._mf_agg_weights(:flow, 3, Float64) == [1.0, 1.0, 1.0]
    @test _M._mf_agg_weights(:average, 3, Float64) ≈ fill(1 / 3, 3)
    # Mariano-Murasawa triangular filter [1,2,3,2,1]/3 over 2m-1 = 5 lags
    @test _M._mf_agg_weights(:growth, 3, Float64) ≈ [1, 2, 3, 2, 1] ./ 3
    @test _M._mf_agg_weights(:growth, 4, Float64) ≈ [1, 2, 3, 4, 3, 2, 1] ./ 4
    @test sum(_M._mf_agg_weights(:growth, 3, Float64)) ≈ 3.0
    @test sum(_M._mf_agg_weights(:average, 5, Float64)) ≈ 1.0

    @test_throws ArgumentError _M._mf_agg_weights(:bogus, 3, Float64)
    @test_throws ArgumentError _M._mf_agg_weights(:flow, 0, Float64)
end

@testset "observation rows at reference and non-reference dates" begin
    n, L, m = 2, 3, 3
    is_low = [false, true]
    weights = [[1.0], _M._mf_agg_weights(:flow, m, Float64)]
    data = [1.0 NaN; 2.0 NaN; 3.0 9.0]

    # A non-reference date sees only the high-frequency series
    Z1, y1 = _M._mf_obs_rows(1, data, is_low, weights, n, L)
    @test size(Z1) == (1, n * L)
    @test y1 == [1.0]
    @test Z1[1, 1] == 1.0 && all(Z1[1, 2:end] .== 0.0)

    # A reference date adds the aggregation row spanning the lag blocks
    Z3, y3 = _M._mf_obs_rows(3, data, is_low, weights, n, L)
    @test size(Z3) == (2, n * L)
    @test y3 == [3.0, 9.0]
    @test Z3[1, 1] == 1.0
    # variable 2 at lags 0,1,2 lives at state positions 2, n+2, 2n+2
    @test Z3[2, 2] == 1.0 && Z3[2, n+2] == 1.0 && Z3[2, 2n+2] == 1.0

    # Nothing observed ⇒ empty system
    Ze, ye = _M._mf_obs_rows(1, fill(NaN, 1, 2), is_low, weights, n, L)
    @test isempty(ye) && size(Ze) == (0, n * L)
end

@testset "companion transition, drift and singular state noise" begin
    n, p, L = 2, 1, 3
    B = [0.1 0.2; 0.6 0.15; 0.1 0.5]        # [c'; A1']  (k × n, k = 1 + n p)
    Sigma = [0.16 0.02; 0.02 0.09]
    Tm, drift, Q = _M._mf_companion(B, Sigma, n, p, L)

    @test size(Tm) == (n * L, n * L)
    @test Tm[1:n, 1:n] == Matrix(B[2:3, :]')      # top block is A₁
    @test Tm[(n+1):(2n), 1:n] == Matrix{Float64}(I, n, n)   # shift register
    @test all(Tm[1:n, (n+1):end] .== 0.0)         # p = 1 ⇒ no higher lags
    @test drift[1:n] == B[1, :]
    @test all(drift[(n+1):end] .== 0.0)
    @test Q[1:n, 1:n] == Sigma
    @test all(Q[(n+1):end, :] .== 0.0)            # noise only in the top block
end

# ─────────────────────────────────────────────────────────────────────────────
# The defining property: the aggregation identity
# ─────────────────────────────────────────────────────────────────────────────

@testset "the latent path reproduces the low-frequency observations exactly" begin
    T_hf = FAST ? 90 : 150
    Z = _mf_sim(Random.MersenneTwister(4), T_hf)
    for kind in (:flow, :average, :growth, :stock)
        m = 3
        data = _mf_blank(Z, 2, kind, m)
        post = estimate_mfvar(data, 1; low_freq=[2], aggregation=kind, freq_ratio=m,
                              n_draws=FAST ? 40 : 80, n_burn=FAST ? 40 : 80,
                              rng=Random.MersenneTwister(3))
        w = _M._mf_agg_weights(kind, m, Float64)
        mu, _ = latent_path(post)

        errs = Float64[]
        for t in 1:T_hf
            isnan(data[t, 2]) && continue
            push!(errs, abs(sum(w[j] * mu[t-j+1, 2] for j in eachindex(w)) - data[t, 2]))
        end
        # The aggregation is noiseless, so the Durbin-Koopman draw satisfies it by
        # construction up to the filter jitter — not merely approximately.
        @test maximum(errs) < 1e-5

        # ... and it holds draw by draw, not just in the posterior mean
        d = 1
        errs_d = Float64[]
        for t in 1:T_hf
            isnan(data[t, 2]) && continue
            push!(errs_d, abs(sum(w[j] * post.Z_draws[d, t-j+1, 2] for j in eachindex(w)) -
                              data[t, 2]))
        end
        @test maximum(errs_d) < 1e-5
    end
end

@testset "high-frequency series pass through untouched" begin
    Z = _mf_sim(Random.MersenneTwister(6), 120)
    data = _mf_blank(Z, 2, :flow, 3)
    post = estimate_mfvar(data, 1; low_freq=[2], aggregation=:flow,
                          n_draws=40, n_burn=40, rng=Random.MersenneTwister(5))
    for d in 1:size(post.Z_draws, 1)
        @test post.Z_draws[d, :, 1] ≈ Z[:, 1] atol = 1e-12
    end
end

@testset "the interpolated path tracks a known latent truth" begin
    T_hf = FAST ? 150 : 240
    Z = _mf_sim(Random.MersenneTwister(1), T_hf)
    data = _mf_blank(Z, 2, :flow, 3)
    post = estimate_mfvar(data, 1; low_freq=[2], aggregation=:flow, freq_ratio=3,
                          n_draws=FAST ? 100 : 200, n_burn=FAST ? 100 : 200,
                          varnames=["m", "q"], rng=Random.MersenneTwister(3))

    mu, qs = latent_path(post)
    @test size(mu) == (T_hf, 2)
    @test size(qs) == (T_hf, 2, 3)
    @test all(qs[:, :, 1] .<= qs[:, :, 2] .<= qs[:, :, 3])

    truth = Z[4:end, 2]
    est = mu[4:end, 2]
    rmse = sqrt(mean((est .- truth) .^ 2))
    @test rmse < 0.8 * std(truth)          # materially better than the unconditional mean
    @test cor(est, truth) > 0.65

    # The bands cover the truth most of the time
    covered = mean(qs[4:end, 2, 1] .<= truth .<= qs[4:end, 2, 3])
    @test covered > 0.5
end

# ─────────────────────────────────────────────────────────────────────────────
# Reduction to the single-frequency BVAR
# ─────────────────────────────────────────────────────────────────────────────

@testset "no low-frequency series reduces to the conjugate BVAR" begin
    Z = _mf_sim(Random.MersenneTwister(9), FAST ? 120 : 180)
    post = estimate_mfvar(Z, 1; n_draws=FAST ? 150 : 300, n_burn=100,
                          rng=Random.MersenneTwister(9))
    bv = estimate_bvar(Z, 1; n_draws=FAST ? 150 : 300, rng=Random.MersenneTwister(9))

    @test isempty(post.low_freq)
    Bm = dropdims(mean(post.B_draws; dims=1); dims=1)
    Bb = dropdims(mean(bv.B_draws; dims=1); dims=1)
    @test Bm ≈ Bb atol = 0.05
    Sm = dropdims(mean(post.Sigma_draws; dims=1); dims=1)
    Sb = dropdims(mean(bv.Sigma_draws; dims=1); dims=1)
    @test Sm ≈ Sb atol = 0.02

    # With nothing latent the state path is exactly the data at every draw
    for d in 1:size(post.Z_draws, 1)
        @test post.Z_draws[d, :, :] ≈ Z atol = 1e-12
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Downstream analysis
# ─────────────────────────────────────────────────────────────────────────────

@testset "forecasts and IRFs at the high frequency" begin
    Z = _mf_sim(Random.MersenneTwister(11), 150)
    data = _mf_blank(Z, 2, :growth, 3)
    post = estimate_mfvar(data, 1; low_freq=[2], aggregation=:growth,
                          n_draws=100, n_burn=80, varnames=["m", "q"],
                          rng=Random.MersenneTwister(13))

    fc = forecast(post, 6)
    @test size(fc.forecast) == (6, 2)
    @test all(isfinite, fc.forecast)
    @test all(isfinite, fc.ci_lower) && all(isfinite, fc.ci_upper)
    @test all(fc.ci_lower .<= fc.forecast .<= fc.ci_upper)

    r = irf(post, 8; method=:cholesky)
    @test size(r.point_estimate) == (8, 2, 2)
    @test all(isfinite, r.point_estimate)
    @test all(isfinite, r.quantiles)
end

# ─────────────────────────────────────────────────────────────────────────────
# Interface
# ─────────────────────────────────────────────────────────────────────────────

@testset "frequency ratios, reproducibility, validation and display" begin
    Z = _mf_sim(Random.MersenneTwister(17), 160)

    # A 4:1 ratio works as well as 3:1
    d4 = _mf_blank(Z, 2, :flow, 4)
    p4 = estimate_mfvar(d4, 1; low_freq=[2], aggregation=:flow, freq_ratio=4,
                        n_draws=40, n_burn=40, rng=Random.MersenneTwister(4))
    @test p4.freq_ratio == 4
    w4 = _M._mf_agg_weights(:flow, 4, Float64)
    mu4, _ = latent_path(p4)
    errs = [abs(sum(w4[j] * mu4[t-j+1, 2] for j in eachindex(w4)) - d4[t, 2])
            for t in 1:160 if !isnan(d4[t, 2])]
    @test maximum(errs) < 1e-5

    data = _mf_blank(Z, 2, :flow, 3)
    a = estimate_mfvar(data, 1; low_freq=[2], n_draws=30, n_burn=30, aggregation=:flow,
                       rng=Random.MersenneTwister(77))
    b = estimate_mfvar(data, 1; low_freq=[2], n_draws=30, n_burn=30, aggregation=:flow,
                       rng=Random.MersenneTwister(77))
    @test a.B_draws == b.B_draws
    @test a.Z_draws == b.Z_draws
    @test _M.n_draws(a) == 30
    @test size(a.B_draws) == (30, 1 + 2 * 1, 2)
    @test size(a.Z_draws) == (30, 160, 2)

    # Per-series aggregation rules
    two_low = copy(Z)
    two_low[:, 1] = _mf_blank(Z, 1, :stock, 3)[:, 1]
    two_low[:, 2] = _mf_blank(Z, 2, :flow, 3)[:, 2]
    pm = estimate_mfvar(two_low, 1; low_freq=[1, 2], aggregation=[:stock, :flow],
                        n_draws=30, n_burn=30, rng=Random.MersenneTwister(19))
    @test pm.aggregation == [:stock, :flow]

    @test_throws ArgumentError estimate_mfvar(data, 0; low_freq=[2])
    @test_throws ArgumentError estimate_mfvar(data, 1; low_freq=[2], n_draws=0)
    @test_throws ArgumentError estimate_mfvar(data, 1; low_freq=[9])
    @test_throws ArgumentError estimate_mfvar(data, 1; low_freq=[2, 2])
    @test_throws ArgumentError estimate_mfvar(data, 1; low_freq=[2], prior=:bogus)
    @test_throws ArgumentError estimate_mfvar(data, 1; low_freq=[2], freq_ratio=0)
    @test_throws ArgumentError estimate_mfvar(data, 1; low_freq=[2],
                                              aggregation=[:flow, :flow])
    @test_throws ArgumentError estimate_mfvar(data, 1; low_freq=[2], varnames=["only"])
    # A high-frequency series containing NaN is a specification error, not silent latency
    @test_throws ArgumentError estimate_mfvar(data, 1; low_freq=Int[])

    out = sprint(show, a)
    @test occursin("Mixed-Frequency VAR", out)
    @test occursin("Schorfheide-Song", out)
    @test occursin("Interpolated high-frequency path", out)
    @test report(a) === nothing

    single = estimate_mfvar(Z, 1; n_draws=20, n_burn=20, rng=Random.MersenneTwister(2))
    @test occursin("single frequency", sprint(show, single))
end

end  # @testset "Mixed-Frequency VAR"
