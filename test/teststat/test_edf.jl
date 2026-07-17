# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-26 (#434): EDF goodness-of-fit battery — KS / Lilliefors / Cramér–von Mises
# (W²) / Anderson–Darling (A²) / Watson (U²) against specified/estimated dists.
#
# Oracle layers:
#  (1) Analytic PIT-formula hand-recomputation (PRIMARY, always available):
#      literal KS D±, CvM W², AD A², Watson U² recomputed inside the test.
#  (2) LIVE cross-implementation (generated offline, exact calls in comments):
#        - scipy.stats.kstest / cramervonmises / anderson  (scipy 1.13.1)
#        - statsmodels.stats.diagnostic.lilliefors         (Dallal–Wilkinson)
#      R nortest/goftest are NOT installed in this env, so KS/CvM/AD specified
#      and estimated-normal AD/CvM/Lilliefors are cross-checked against scipy /
#      statsmodels instead (equivalent numerics).
#  (3) PUBLISHED-with-citation: the estimated-normal AD/CvM p-value closed forms
#      (D'Agostino & Stephens 1986) and Case-3 critical values (Stephens 1974).
#  (4) Property / seeded Monte-Carlo calibration + clean degenerate-input errors.

using Test, MacroEconometricModels, Random, Statistics, Distributions

# --- Fixed oracle vectors (values generated offline, pasted verbatim) --------
# VEC = numpy default_rng(434).normal(0,1,40) rounded to 6 dp.
const EDF_VEC = [-1.601983, 0.859928, -2.268961, -1.748823, 0.066485, 1.121189,
    1.991146, 1.145641, 1.498317, -1.469578, -1.60543, -0.498326, 0.278593,
    -1.741057, -0.134895, -0.304083, 0.213324, 0.558596, -0.810896, -0.571091,
    0.516309, 0.088412, 1.449091, 0.274737, 0.347131, 1.114875, 1.804228,
    0.417866, -0.534543, 1.401187, -0.460337, 0.561794, 2.787116, 0.763495,
    1.490419, -1.464275, -0.700287, -0.529766, 2.226565, -0.41889]
# XREJ = numpy default_rng(77).exponential(1,30) rounded to 6 dp (non-normal).
const EDF_XREJ = [2.397793, 0.795045, 0.139325, 0.864483, 0.922846, 0.272859,
    3.028332, 0.274575, 1.845248, 0.743228, 3.523643, 1.203363, 0.256742,
    0.364885, 1.348096, 2.307059, 0.77763, 0.412592, 2.253151, 0.202599,
    0.90534, 0.266732, 1.466933, 0.538642, 0.434289, 2.961536, 0.169849,
    0.607992, 0.001259, 1.502599]

# Independent literal re-implementation of the four EDF statistics from sorted z.
function _edf_hand(z)
    n = length(z)
    i = 1:n
    dplus  = maximum(i ./ n .- z)
    dminus = maximum(z .- (i .- 1) ./ n)
    D = max(dplus, dminus)
    W2 = 1 / (12n) + sum((z .- (2 .* i .- 1) ./ (2n)) .^ 2)
    zc = clamp.(z, eps(Float64), 1 - eps(Float64))
    A2 = -n - (1 / n) * sum((2 .* i .- 1) .* (log.(zc) .+ log.(1 .- reverse(zc))))
    U2 = W2 - n * (mean(z) - 0.5)^2
    return (; D, dplus, dminus, W2, A2, U2)
end

@testset "EDF goodness-of-fit battery (EV-26)" begin

    @testset "Analytic PIT-formula hand-recomputation (primary)" begin
        # Tiny sorted vector; specified N(0,1) so z = Φ(y).
        y = [-1.2, -0.3, 0.1, 0.7, 1.5, 2.1]
        z = sort(cdf.(Normal(0, 1), y))
        h = _edf_hand(z)
        # KS
        r = edf_test(y; dist=:normal, test=:ks, params=:specified, theta=(0.0, 1.0))
        @test r.raw_statistic ≈ h.D atol=1e-12
        @test max(h.dplus, h.dminus) ≈ h.D
        # CvM
        r = edf_test(y; dist=:normal, test=:cvm, params=:specified, theta=(0.0, 1.0))
        @test r.raw_statistic ≈ h.W2 atol=1e-12
        # AD
        r = edf_test(y; dist=:normal, test=:ad, params=:specified, theta=(0.0, 1.0))
        @test r.raw_statistic ≈ h.A2 atol=1e-12
        # Watson: identity U² = W² − n(z̄−0.5)²
        r = edf_test(y; dist=:normal, test=:watson, params=:specified, theta=(0.0, 1.0))
        @test r.raw_statistic ≈ h.U2 atol=1e-12
        @test r.raw_statistic ≈ h.W2 - length(z) * (mean(z) - 0.5)^2 atol=1e-12
    end

    @testset "AD ln-clamp guards against Inf at boundary PITs" begin
        # Extreme values push Φ to 0 / 1; A² must stay finite, never ±Inf/NaN.
        y = [-40.0, -20.0, 0.0, 20.0, 40.0]
        r = edf_test(y; dist=:normal, test=:ad, params=:specified, theta=(0.0, 1.0))
        @test isfinite(r.statistic)
        @test !isnan(r.statistic)
    end

    @testset "KS specified — scipy oracle (LIVE) + MTW exact" begin
        # scipy.stats.kstest(EDF_VEC, 'norm')  -> D=0.14254800..., p=0.35650975...
        r = edf_test(EDF_VEC; dist=:normal, test=:ks, params=:specified, theta=(0.0, 1.0))
        @test r.statistic ≈ 0.14254800439512738 atol=1e-8
        @test r.pvalue ≈ 0.35650975465617996 atol=1e-6   # MTW exact matches scipy
        # Hand-recompute matches too.
        h = _edf_hand(sort(cdf.(Normal(0, 1), EDF_VEC)))
        @test r.statistic ≈ h.D atol=1e-12
    end

    @testset "CvM specified — scipy oracle (LIVE, statistic)" begin
        # scipy.stats.cramervonmises(EDF_VEC, 'norm').statistic = 0.17318175...
        r = edf_test(EDF_VEC; dist=:normal, test=:cvm, params=:specified, theta=(0.0, 1.0))
        @test r.statistic ≈ 0.17318175350250598 atol=1e-8
        @test 0.0 < r.pvalue < 1.0
    end

    @testset "AD specified — hand + ADinf asymptotic" begin
        # A² hand-recomputed = 1.35110607 (matches literal formula).
        r = edf_test(EDF_VEC; dist=:normal, test=:ad, params=:specified, theta=(0.0, 1.0))
        @test r.statistic ≈ 1.3511060672883133 atol=1e-8
        # Marsaglia–Marsaglia (2004) ADinf asymptotic p-value.
        @test r.pvalue ≈ 0.2163931051875937 atol=1e-6
    end

    @testset "Watson specified — hand oracle" begin
        r = edf_test(EDF_VEC; dist=:normal, test=:watson, params=:specified, theta=(0.0, 1.0))
        @test r.statistic ≈ 0.11059971784945323 atol=1e-8
    end

    @testset "Estimated normal — Anderson–Darling (scipy anderson, LIVE)" begin
        # scipy.stats.anderson(EDF_VEC, 'norm').statistic = 0.22608181 (uncorrected A²,
        # sample-sd standardization). We store it as raw_statistic.
        r = edf_test(EDF_VEC; dist=:normal, test=:ad, params=:estimate)
        @test r.raw_statistic ≈ 0.22608181267905536 atol=1e-6
        # Stephens (1974) modification A²* = A²(1+0.75/n+2.25/n²).
        n = length(EDF_VEC)
        @test r.statistic ≈ 0.22608181267905536 * (1 + 0.75/n + 2.25/n^2) atol=1e-8
        # D'Agostino & Stephens (1986) closed-form p-value (published).
        @test r.pvalue ≈ 0.8052 atol=2e-3
        @test r.params == :estimate
        @test occursin("Case 3", r.case)
        # A clearly non-normal sample must reject:
        # scipy.stats.anderson(EDF_XREJ,'norm').statistic = 1.44011220
        r2 = edf_test(EDF_XREJ; dist=:normal, test=:ad, params=:estimate)
        @test r2.raw_statistic ≈ 1.4401121958115013 atol=1e-5
        @test r2.pvalue < 0.01
    end

    @testset "Estimated normal — Cramér–von Mises (statistic + Stephens mod)" begin
        # Raw W² (sample-sd standardization) hand-value = 0.02881395.
        r = edf_test(EDF_VEC; dist=:normal, test=:cvm, params=:estimate)
        @test r.raw_statistic ≈ 0.028813951429759826 atol=1e-7
        n = length(EDF_VEC)
        @test r.statistic ≈ 0.028813951429759826 * (1 + 0.5/n) atol=1e-8
        @test isfinite(r.pvalue) && 0.0 < r.pvalue <= 1.0
    end

    @testset "Estimated normal — Lilliefors (statsmodels oracle, LIVE)" begin
        # statsmodels.stats.diagnostic.lilliefors(EDF_VEC, dist='norm'):
        #   D = 0.08344456,  p ≈ 0.68 (large; table method)
        r = edf_test(EDF_VEC; dist=:normal, test=:lilliefors, params=:estimate)
        @test r.statistic ≈ 0.08344456063952205 atol=1e-6
        @test r.pvalue > 0.10                       # correct qualitative region
        @test occursin("Lilliefors", r.case)
        # Rejecting sample in the Dallal–Wilkinson valid range (p<0.10):
        #   lilliefors(EDF_XREJ,'norm',pvalmethod='approx') -> D=0.20268875, p=0.00286448
        r2 = edf_test(EDF_XREJ; dist=:normal, test=:lilliefors, params=:estimate)
        @test r2.statistic ≈ 0.20268875036667733 atol=1e-6
        @test r2.pvalue ≈ 0.002864484381120227 atol=1e-6   # exact D-W match
        # test=:ks with params=:estimate on a normal null routes to the same Lilliefors p.
        rks = edf_test(EDF_XREJ; dist=:normal, test=:ks, params=:estimate)
        @test rks.pvalue ≈ r2.pvalue atol=1e-12
    end

    @testset "Estimated vs specified routes DIFFER (Stephens modification)" begin
        # Same data & statistic family, different p-value machinery.
        r_spec = edf_test(EDF_VEC; dist=:normal, test=:ad, params=:specified, theta=(0.0, 1.0))
        r_est  = edf_test(EDF_VEC; dist=:normal, test=:ad, params=:estimate)
        @test r_spec.pvalue != r_est.pvalue
        @test r_spec.statistic != r_est.statistic     # raw vs modified/refit
        @test occursin("Case 0", r_spec.case)
        @test occursin("Case 3", r_est.case)
    end

    @testset "Every (dist, test, params) combination runs and renders" begin
        pos = abs.(EDF_VEC) .+ 0.05           # positive-support families
        specified_theta = Dict(
            :normal => (0.0, 1.0), :exponential => (1.0,), :logistic => (0.0, 1.0),
            :gumbel => (0.0, 1.0), :gamma => (2.0, 1.0), :weibull => (1.5, 1.0),
            :chisq => (3.0,))
        for dist in (:normal, :exponential, :logistic, :gumbel, :gamma, :weibull, :chisq)
            data = dist in (:exponential, :gamma, :weibull, :chisq) ? pos : EDF_VEC
            for test in (:ks, :cvm, :ad, :watson)
                # estimated route
                re = edf_test(data; dist=dist, test=test, params=:estimate)
                @test re isa EDFTestResult
                @test isfinite(re.statistic)
                @test MacroEconometricModels.StatsAPI.nobs(re) == length(data)
                # renders without error
                @test (io = IOBuffer(); show(io, re); occursin("EDF Goodness-of-Fit", String(take!(io))))
                # non-normal estimated => no null table => NaN p-value + note
                if dist != :normal
                    @test isnan(re.pvalue)
                    @test occursin("no published null table", re.case)
                end
                # specified route (distribution-free asymptotics, always a p-value)
                rs = edf_test(data; dist=dist, test=test, params=:specified,
                              theta=specified_theta[dist])
                @test isfinite(rs.pvalue)
                @test 0.0 <= rs.pvalue <= 1.0
                @test !isempty(rs.critical_values)
            end
        end
    end

    @testset "StatsAPI + refs + report integration" begin
        r = edf_test(EDF_VEC; dist=:normal, test=:ad, params=:estimate)
        @test MacroEconometricModels.StatsAPI.pvalue(r) === r.pvalue
        @test MacroEconometricModels.StatsAPI.nobs(r) == length(EDF_VEC)
        @test (io = IOBuffer(); refs(io, r); occursin("Stephens", String(take!(io))))
        @test refs(devnull, r) === nothing
        @test (redirect_stdout(devnull) do; report(r); end; true)   # report writes to stdout
    end

    @testset "Property: seeded MC calibration of KS specified" begin
        # N(0,1) data tested against N(0,1) specified: PIT is Uniform(0,1); the
        # KS rejection rate at 5% should sit near nominal (loose MC band).
        rng = MersenneTwister(20260717)
        reps = 600
        nrej = 0
        for _ in 1:reps
            y = randn(rng, 60)
            r = edf_test(y; dist=:normal, test=:ks, params=:specified, theta=(0.0, 1.0))
            r.pvalue < 0.05 && (nrej += 1)
        end
        rate = nrej / reps
        @test 0.02 <= rate <= 0.10          # loose band around 0.05
    end

    @testset "Degenerate / invalid input errors cleanly" begin
        @test_throws ArgumentError edf_test(fill(3.0, 25))                 # constant
        @test_throws ArgumentError edf_test(randn(MersenneTwister(1), 3))  # n < 5
        @test_throws ArgumentError edf_test(randn(MersenneTwister(1), 20); dist=:bogus)
        @test_throws ArgumentError edf_test(randn(MersenneTwister(1), 20); test=:bogus)
        @test_throws ArgumentError edf_test(randn(MersenneTwister(1), 20); params=:bogus)
        # specified without theta
        @test_throws ArgumentError edf_test(randn(MersenneTwister(1), 20); params=:specified)
        # lilliefors only for normal + estimate
        @test_throws ArgumentError edf_test(abs.(EDF_VEC) .+ 0.1; dist=:exponential, test=:lilliefors)
        @test_throws ArgumentError edf_test(EDF_VEC; test=:lilliefors, params=:specified, theta=(0.0, 1.0))
        # positive-support family with non-positive data
        @test_throws ArgumentError edf_test(EDF_VEC; dist=:exponential, params=:estimate)
    end

    @testset "Integer / abstract vector fallback" begin
        yi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6]
        r = edf_test(yi; dist=:normal, test=:ad, params=:estimate)
        @test r isa EDFTestResult{Float64}
        @test isfinite(r.statistic)
    end
end
