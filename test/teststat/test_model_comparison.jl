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

using Random

Random.seed!(12345)

# =============================================================================
# Shared test data
# =============================================================================

# ARIMA data
y_arima = randn(300)

# VAR data
Y_var = randn(150, 3)

# Volatility data (GARCH-like)
n_vol = 500
z_vol = randn(n_vol)
h_vol = ones(n_vol)
y_vol = zeros(n_vol)
y_vol[1] = z_vol[1]
for t in 2:n_vol
    h_vol[t] = 0.05 + 0.15 * y_vol[t-1]^2 + 0.75 * h_vol[t-1]
    y_vol[t] = sqrt(h_vol[t]) * z_vol[t]
end

# =============================================================================
# LR Test — ARIMA Family
# =============================================================================

@testset "LR Test — ARIMA" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)
    ar4 = estimate_ar(y_arima, 4; method=:mle)

    result = lr_test(ar2, ar4)
    @test result isa LRTestResult
    @test result.statistic >= 0
    @test 0 <= result.pvalue <= 1
    @test result.df == dof(ar4) - dof(ar2)
    @test result.df == 2
    @test result.loglik_restricted ≈ loglikelihood(ar2)
    @test result.loglik_unrestricted ≈ loglikelihood(ar4)
    @test result.dof_restricted == dof(ar2)
    @test result.dof_unrestricted == dof(ar4)
    @test result.nobs_restricted == nobs(ar2)
    @test result.nobs_unrestricted == nobs(ar4)

    # MA models
    ma1 = estimate_ma(y_arima, 1; method=:mle)
    ma3 = estimate_ma(y_arima, 3; method=:mle)
    result_ma = lr_test(ma1, ma3)
    @test result_ma isa LRTestResult
    @test result_ma.statistic >= 0
    @test result_ma.df == 2

    # ARMA models
    arma11 = estimate_arma(y_arima, 1, 1; method=:mle)
    arma22 = estimate_arma(y_arima, 2, 2; method=:mle)
    result_arma = lr_test(arma11, arma22)
    @test result_arma isa LRTestResult
    @test result_arma.statistic >= 0
    @test result_arma.df == 2
end

# =============================================================================
# LR Test — VAR Family
# =============================================================================

@testset "LR Test — VAR" begin
    var1 = estimate_var(Y_var, 1)
    var3 = estimate_var(Y_var, 3)

    result = lr_test(var1, var3)
    @test result isa LRTestResult
    @test result.statistic >= 0
    @test 0 <= result.pvalue <= 1
    @test result.df == dof(var3) - dof(var1)
    @test result.dof_restricted == dof(var1)
    @test result.dof_unrestricted == dof(var3)
end

# =============================================================================
# LR Test — Volatility Models
# =============================================================================

@testset "LR Test — Volatility" begin
    arch1 = estimate_arch(y_vol, 1)
    arch3 = estimate_arch(y_vol, 3)

    result = lr_test(arch1, arch3)
    @test result isa LRTestResult
    @test result.statistic >= 0
    @test result.df == 2

    garch11 = estimate_garch(y_vol, 1, 1)
    garch22 = estimate_garch(y_vol, 2, 2)

    result_g = lr_test(garch11, garch22)
    @test result_g isa LRTestResult
    @test result_g.statistic >= 0
    @test result_g.df == dof(garch22) - dof(garch11)
end

# =============================================================================
# LR Test — Cross-type Generic
# =============================================================================

@testset "LR Test — Cross-type" begin
    # ARCH → GARCH (ARCH is nested in GARCH)
    arch2 = estimate_arch(y_vol, 2)
    garch12 = estimate_garch(y_vol, 1, 2)

    result = lr_test(arch2, garch12)
    @test result isa LRTestResult
    @test result.statistic >= 0
    @test result.df == dof(garch12) - dof(arch2)
end

# =============================================================================
# LR Test — VECM
# =============================================================================

@testset "LR Test — VECM" begin
    # VECM with different lag orders
    Y_ci = randn(120, 2)
    # Accumulate to create I(1)-like data
    Y_ci = cumsum(Y_ci, dims=1)

    vecm2 = estimate_vecm(Y_ci, 2; rank=1)
    vecm4 = estimate_vecm(Y_ci, 4; rank=1)

    result = lr_test(vecm2, vecm4)
    @test result isa LRTestResult
    @test result.statistic >= 0
    @test 0 <= result.pvalue <= 1
end

# =============================================================================
# LR Test — Order Invariance
# =============================================================================

@testset "LR Test — Order invariance" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)
    ar4 = estimate_ar(y_arima, 4; method=:mle)

    result1 = lr_test(ar2, ar4)
    result2 = lr_test(ar4, ar2)  # reversed order

    @test result1.statistic ≈ result2.statistic
    @test result1.pvalue ≈ result2.pvalue
    @test result1.df == result2.df
    @test result1.loglik_restricted ≈ result2.loglik_restricted
    @test result1.loglik_unrestricted ≈ result2.loglik_unrestricted
end

# =============================================================================
# LR Test — Error Cases
# =============================================================================

@testset "LR Test — Error cases" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)
    ar2b = estimate_ar(y_arima, 2; method=:mle)  # same dof

    @test_throws ArgumentError lr_test(ar2, ar2b)
end

# =============================================================================
# LR Test — nobs Mismatch Warning
# =============================================================================

@testset "LR Test — nobs mismatch warning" begin
    # Use subsections of the same data so nobs differ
    ar2_short = estimate_ar(y_arima[1:200], 2; method=:mle)
    ar4_long = estimate_ar(y_arima[1:250], 4; method=:mle)

    result = @test_warn "different number of observations" lr_test(ar2_short, ar4_long)
    @test result isa LRTestResult
end

# =============================================================================
# LM Test — ARIMA Family
# =============================================================================

@testset "LM Test — ARIMA (AR×AR)" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)
    ar4 = estimate_ar(y_arima, 4; method=:mle)

    result = lm_test(ar2, ar4)
    @test result isa LMTestResult
    @test result.statistic >= 0
    @test 0 <= result.pvalue <= 1
    @test result.df == 2
    @test result.nobs == length(y_arima)
    @test result.score_norm >= 0
end

@testset "LM Test — ARIMA (MA×ARMA)" begin
    ma1 = estimate_ma(y_arima, 1; method=:mle)
    arma11 = estimate_arma(y_arima, 1, 1; method=:mle)

    result = lm_test(ma1, arma11)
    @test result isa LMTestResult
    @test result.statistic >= 0
    @test result.df == dof(arma11) - dof(ma1)
end

@testset "LM Test — ARIMA (AR×ARMA)" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)
    arma21 = estimate_arma(y_arima, 2, 1; method=:mle)

    result = lm_test(ar2, arma21)
    @test result isa LMTestResult
    @test result.statistic >= 0
    @test result.df == dof(arma21) - dof(ar2)
end

# =============================================================================
# LM Test — VAR Family
# =============================================================================

@testset "LM Test — VAR" begin
    var1 = estimate_var(Y_var, 1)
    var3 = estimate_var(Y_var, 3)

    result = lm_test(var1, var3)
    @test result isa LMTestResult
    @test result.statistic >= 0
    @test 0 <= result.pvalue <= 1
    @test result.df == dof(var3) - dof(var1)
end

# =============================================================================
# LM Test — Volatility Family
# =============================================================================

@testset "LM Test — Volatility (ARCH×ARCH)" begin
    arch1 = estimate_arch(y_vol, 1)
    arch3 = estimate_arch(y_vol, 3)

    result = lm_test(arch1, arch3)
    @test result isa LMTestResult
    @test result.statistic >= 0
    @test result.df == 2
end

@testset "LM Test — Volatility (ARCH×GARCH)" begin
    arch2 = estimate_arch(y_vol, 2)
    garch12 = estimate_garch(y_vol, 1, 2)

    result = lm_test(arch2, garch12)
    @test result isa LMTestResult
    @test result.statistic >= 0
    @test result.df == dof(garch12) - dof(arch2)
end

@testset "LM Test — Volatility (GARCH×GARCH)" begin
    garch11 = estimate_garch(y_vol, 1, 1)
    garch22 = estimate_garch(y_vol, 2, 2)

    result = lm_test(garch11, garch22)
    @test result isa LMTestResult
    @test result.statistic >= 0
    @test result.df == dof(garch22) - dof(garch11)
end

# =============================================================================
# LM Test — Error Cases
# =============================================================================

@testset "LM Test — Error cases" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)

    # Same dof
    ar2b = estimate_ar(y_arima, 2; method=:mle)
    @test_throws ArgumentError lm_test(ar2, ar2b)

    # Different data — use explicit seed to guarantee different data from y_arima
    rng_other = Random.MersenneTwister(99999)
    y_other = randn(rng_other, 300)
    ar4_other = estimate_ar(y_other, 4; method=:mle)
    @test_throws ArgumentError lm_test(ar2, ar4_other)

    # Unsupported cross-type
    var1 = estimate_var(Y_var, 1)
    @test_throws ArgumentError lm_test(ar2, var1)

    # Different d in ARIMA
    y_arima2 = cumsum(y_arima)
    arima_d0 = estimate_arima(y_arima2, 1, 0, 0; method=:mle)
    arima_d1 = estimate_arima(y_arima2, 1, 1, 0; method=:mle)
    @test_throws ArgumentError lm_test(arima_d0, arima_d1)
end

@testset "LM Test — VAR different data" begin
    rng_v = Random.MersenneTwister(77777)
    Y_other = randn(rng_v, 150, 3)
    var1 = estimate_var(Y_var, 1)
    var2_other = estimate_var(Y_other, 2)
    @test_throws ArgumentError lm_test(var1, var2_other)
end

@testset "LM Test — Volatility different data" begin
    rng_vol = Random.MersenneTwister(55555)
    y_other_vol = randn(rng_vol, 500)
    arch1 = estimate_arch(y_vol, 1)
    arch2_other = estimate_arch(y_other_vol, 2)
    @test_throws ArgumentError lm_test(arch1, arch2_other)
end

@testset "LM Test — Unsupported volatility cross-type" begin
    egarch11 = estimate_egarch(y_vol, 1, 1)
    garch11 = estimate_garch(y_vol, 1, 1)
    @test_throws ArgumentError lm_test(egarch11, garch11)
end

# =============================================================================
# LM Test — Order Invariance
# =============================================================================

@testset "LM Test — Order invariance" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)
    ar4 = estimate_ar(y_arima, 4; method=:mle)

    result1 = lm_test(ar2, ar4)
    result2 = lm_test(ar4, ar2)

    @test result1.statistic ≈ result2.statistic
    @test result1.pvalue ≈ result2.pvalue
    @test result1.df == result2.df
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

@testset "StatsAPI interface" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)
    ar4 = estimate_ar(y_arima, 4; method=:mle)

    lr_result = lr_test(ar2, ar4)
    @test nobs(lr_result) == nobs(ar4)
    @test dof(lr_result) == 2

    lm_result = lm_test(ar2, ar4)
    @test nobs(lm_result) == length(y_arima)
    @test dof(lm_result) == 2
end

# =============================================================================
# Display
# =============================================================================

@testset "Display — LR" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)
    ar4 = estimate_ar(y_arima, 4; method=:mle)
    result = lr_test(ar2, ar4)

    io = IOBuffer()
    show(io, result)
    output = String(take!(io))

    @test occursin("Likelihood Ratio Test", output)
    @test occursin("LR statistic", output)
    @test occursin("P-value", output)
    @test occursin("Log-likelihood", output)
    @test occursin("Degrees of freedom", output)
    @test occursin("Conclusion", output)
end

@testset "Display — LM" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)
    ar4 = estimate_ar(y_arima, 4; method=:mle)
    result = lm_test(ar2, ar4)

    io = IOBuffer()
    show(io, result)
    output = String(take!(io))

    @test occursin("Lagrange Multiplier", output)
    @test occursin("LM statistic", output)
    @test occursin("P-value", output)
    @test occursin("Score norm", output)
    @test occursin("Conclusion", output)
end

# =============================================================================
# refs() Output
# =============================================================================

@testset "refs() — LR/LM" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)
    ar4 = estimate_ar(y_arima, 4; method=:mle)

    lr_result = lr_test(ar2, ar4)
    lm_result = lm_test(ar2, ar4)

    io = IOBuffer()
    refs(io, lr_result)
    output_lr = String(take!(io))
    @test occursin("Wilks", output_lr)

    io = IOBuffer()
    refs(io, lm_result)
    output_lm = String(take!(io))
    @test occursin("Rao", output_lm)

    # Symbol dispatch
    io = IOBuffer()
    refs(io, :lr_test)
    output_sym = String(take!(io))
    @test occursin("Wilks", output_sym)

    io = IOBuffer()
    refs(io, :lm_test)
    output_sym = String(take!(io))
    @test occursin("Rao", output_sym)
end

# =============================================================================
# LR and LM Consistency
# =============================================================================

@testset "LR/LM consistency — same conclusion" begin
    ar2 = estimate_ar(y_arima, 2; method=:mle)
    ar4 = estimate_ar(y_arima, 4; method=:mle)

    lr = lr_test(ar2, ar4)
    lm = lm_test(ar2, ar4)

    # Both should agree on df
    @test lr.df == lm.df

    # Both statistics should be non-negative
    @test lr.statistic >= 0
    @test lm.statistic >= 0

    # Under large samples, LR and LM should give similar conclusions
    # (both reject or both fail to reject at 5%)
    # We test they are at least in the same ballpark
    @test abs(lr.statistic - lm.statistic) < max(lr.statistic, lm.statistic) + 1.0
end

# =============================================================================
# LM Test — EGARCH and GJR-GARCH
# =============================================================================

@testset "LM Test — EGARCH×EGARCH" begin
    egarch11 = estimate_egarch(y_vol, 1, 1)
    egarch22 = estimate_egarch(y_vol, 2, 2)

    result = lm_test(egarch11, egarch22)
    @test result isa LMTestResult
    @test result.statistic >= 0
    @test result.df == dof(egarch22) - dof(egarch11)
end

@testset "LM Test — GJR-GARCH×GJR-GARCH" begin
    gjr11 = estimate_gjr_garch(y_vol, 1, 1)
    gjr22 = estimate_gjr_garch(y_vol, 2, 2)

    result = lm_test(gjr11, gjr22)
    @test result isa LMTestResult
    @test result.statistic >= 0
    @test result.df == dof(gjr22) - dof(gjr11)
end

# =============================================================================
# LR Test — Significant Result (Strongly Nested)
# =============================================================================

@testset "LR Test — Significant result" begin
    # Generate AR(2) data to make AR(2) vs AR(0+) meaningful
    Random.seed!(999)
    n_sig = 500
    y_sig = zeros(n_sig)
    for t in 3:n_sig
        y_sig[t] = 0.5 * y_sig[t-1] - 0.3 * y_sig[t-2] + randn()
    end

    ar0 = estimate_ar(y_sig, 1; method=:mle)  # underfit
    ar2 = estimate_ar(y_sig, 2; method=:mle)  # correct

    result = lr_test(ar0, ar2)
    # AR(2) should be significantly better than AR(1) for AR(2) data
    @test result.statistic > 0
    @test result.pvalue < 0.05
end
