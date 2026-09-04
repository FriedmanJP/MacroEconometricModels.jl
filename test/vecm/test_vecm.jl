# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test, MacroEconometricModels, Random, LinearAlgebra, Statistics

# =============================================================================
# DGP-04 (#793): parametrised VECM truths on the shared simulator.
# The old corner fixture (instant full adjustment α = −1, no Γ, every extra
# column loading on the SAME trend) is retired: adjustment speeds are moderate,
# short-run dynamics are non-zero, and rank-2 trends are distinct — so rank
# selection, (α, β, Γ) recovery and causal direction are all testable.
# NOTE: `make_cointegrated_data` (test/fixtures.jl) stays until its remaining
# callers migrate (display/coverage, DGP-18 #807).
# =============================================================================

const _VECM_A1 = [-0.3, 0.1, 0.0]   # rank-1 adjustment speeds
const _VECM_B1 = [1.0, -1.0, 0.0]   # rank-1 cointegrating vector

"""Rank-1 3-variable VECM truth (default design: α, β, Γ = 0.2I)."""
_vecm_dgp(rng::AbstractRNG, T::Int=400; kwargs...) = dgp_vecm(rng; T=T, kwargs...).Y

"""Rank-1 bivariate VECM truth."""
_vecm_biv(rng::AbstractRNG, T::Int=200) =
    dgp_vecm(rng; alpha=[-0.3, 0.1], beta=[1.0, -1.0],
             Gamma=Matrix(0.2 * I, 2, 2), T=T).Y

"""Rank-2 4-variable truth: two independent cointegrated pairs."""
const _VECM_A2 = [-0.4 0.0; 0.0 -0.4; 0.2 0.0; 0.0 0.2]
const _VECM_B2 = [1.0 0.0; 0.0 1.0; -1.0 0.0; 0.0 -1.0]
_vecm_dgp2(rng::AbstractRNG, T::Int=400) =
    dgp_vecm(rng; alpha=_VECM_A2, beta=_VECM_B2,
             Gamma=Matrix(0.2 * I, 4, 4), T=T).Y

"""Rank-0 truth: independent random walks (no cointegration)."""
_vecm_rw(rng::AbstractRNG, T::Int=400) =
    dgp_vecm(rng; alpha=zeros(3, 0), beta=zeros(3, 0),
             Gamma=zeros(3, 3), T=T).Y

"""Directional truth: only equation 1 error-corrects (known causal direction)."""
_vecm_directional(rng::AbstractRNG, T::Int=400) =
    dgp_vecm(rng; alpha=[-0.3, 0.0, 0.0], T=T).Y

"""Drift truth with β′μ = 0 (drift along the attractor; cointegration stays clean)."""
_vecm_drift(rng::AbstractRNG, T::Int=400) =
    dgp_vecm(rng; mu=[0.05, 0.05, 0.05], T=T).Y

# =============================================================================
# Johansen Estimation
# =============================================================================

@testset "VECM Johansen Estimation" begin

    @testset "Basic estimation" begin
        Y = _vecm_dgp(MersenneTwister(42), 200)
        m = estimate_vecm(Y, 2)

        @test m isa VECMModel{Float64}
        @test nvars(m) == 3
        @test nlags(m) == 2
        @test m.p == 2
        @test m.method == :johansen
        @test m.deterministic == :constant
        @test effective_nobs(m) > 0
        @test effective_nobs(m) == size(m.U, 1)
        @test size(m.U, 2) == 3
        @test size(m.Sigma) == (3, 3)
        @test issymmetric(round.(m.Sigma, digits=10))
        @test isfinite(m.aic)
        @test isfinite(m.bic)
        @test isfinite(m.hqic)
        @test isfinite(m.loglik)
    end

    @testset "Rank detection" begin
        # Rank selection recovers the true rank (DGP-04 #793) — both criteria,
        # on rank-0/1/2 truths. Seeds are calibrated: selection is a 5%-level
        # test, so an exact `==` must sit on a non-marginal draw (verified on
        # Julia 1.12; MT streams are version-stable across Julia versions).
        Y1 = _vecm_dgp(MersenneTwister(11), 400)
        @test select_vecm_rank(Y1, 2; criterion=:trace) == 1
        @test select_vecm_rank(Y1, 2; criterion=:max_eigen) == 1
        Y0 = _vecm_rw(MersenneTwister(12), 400)
        @test select_vecm_rank(Y0, 2; criterion=:trace) == 0
        @test select_vecm_rank(Y0, 2; criterion=:max_eigen) == 0
        Y2 = _vecm_dgp2(MersenneTwister(14), 400)
        @test select_vecm_rank(Y2, 2; criterion=:trace) == 2
        @test select_vecm_rank(Y2, 2; criterion=:max_eigen) == 2

        # The default estimator path auto-selects rank 1 on the rank-1 truth
        m = estimate_vecm(Y1, 2)
        @test m.rank == 1
        @test size(m.alpha) == (3, m.rank)
        @test size(m.beta) == (3, m.rank)
        @test size(m.Pi) == (3, 3)

        # Explicit rank
        m2 = estimate_vecm(Y1, 2; rank=1)
        @test m2.rank == 1
        @test size(m2.alpha) == (3, 1)
        @test size(m2.beta) == (3, 1)
    end

    @testset "(α, β, Γ) recovery on the rank-1 truth" begin
        # Phillips normalization pins β[1, 1] = 1; the rest is estimated.
        # Bounds carry a ≥2x margin over seeds 11/21/31 (β ≤ 0.02, α ≤ 0.04,
        # Γ ≤ 0.083 there) — the #258 one-line rationale.
        d = dgp_vecm(MersenneTwister(11); T=400)
        m = estimate_vecm(d.Y, 2; rank=1)
        @test maximum(abs, vec(m.beta) - _VECM_B1) < 0.05
        @test maximum(abs, vec(m.alpha) - _VECM_A1) < 0.1
        @test maximum(abs, m.Gamma[1] - 0.2 * I) < 0.15
    end

    @testset "Rank 2 system" begin
        Y = _vecm_dgp2(MersenneTwister(14), 400)
        m = estimate_vecm(Y, 2; rank=2)
        @test m.rank == 2
        @test size(m.alpha) == (4, 2)
        @test size(m.beta) == (4, 2)
        @test size(m.Pi) == (4, 4)
    end

    @testset "(α, β) recovery on the rank-2 truth" begin
        # Seed 13 deliberately: selection over-rejects there (a 5% false
        # rejection), which must NOT affect recovery at an explicit rank.
        m = estimate_vecm(_vecm_dgp2(MersenneTwister(13), 400), 2; rank=2)
        # Phillips pins the first r rows to I; rows 3:4 are comparable.
        @test maximum(abs, m.beta[3:4, :] - _VECM_B2[3:4, :]) < 0.15
        @test maximum(abs, m.alpha - _VECM_A2) < 0.15
    end

    @testset "Deterministic specifications" begin
        Y = _vecm_dgp(MersenneTwister(42), 200)

        for det in (:none, :constant, :trend)
            m = estimate_vecm(Y, 2; rank=1, deterministic=det)
            @test m.deterministic == det
            @test m isa VECMModel{Float64}
        end

        # A DGP WITH drift separates the specifications (DGP-04 #793): the old
        # trendless fixture made :none/:constant/:trend all correct at once.
        # m.mu cannot recover the drift point-wise (I(1) contamination through
        # β̂ error), so assert what the constant honestly delivers: nesting and
        # the exact OLS fitted-mean identity.
        Yd = _vecm_drift(MersenneTwister(16), 400)
        mc = estimate_vecm(Yd, 2; rank=1, deterministic=:constant)
        mn = estimate_vecm(Yd, 2; rank=1, deterministic=:none)
        @test mc.loglik >= mn.loglik   # nested models: constant can only fit better
        @test vec(sum(residuals(mc); dims=1)) ≈ zeros(3) atol = 1e-8
        @test any(abs.(vec(sum(residuals(mn); dims=1))) .> 1e-8)  # no const: drift leaks
    end

    @testset "Different lag orders" begin
        Y = _vecm_dgp(MersenneTwister(42), 200)

        m1 = estimate_vecm(Y, 1; rank=1)
        @test m1.p == 1
        @test isempty(m1.Gamma)

        m2 = estimate_vecm(Y, 2; rank=1)
        @test m2.p == 2
        @test length(m2.Gamma) == 1
        @test size(m2.Gamma[1]) == (3, 3)

        m3 = estimate_vecm(Y, 3; rank=1)
        @test m3.p == 3
        @test length(m3.Gamma) == 2
    end

    @testset "Pi = alpha * beta'" begin
        Y = _vecm_dgp(MersenneTwister(42), 200)
        m = estimate_vecm(Y, 2; rank=1)
        @test m.Pi ≈ m.alpha * m.beta' atol=1e-10
    end

    @testset "Phillips normalization" begin
        Y = _vecm_dgp(MersenneTwister(42), 200)
        m = estimate_vecm(Y, 2; rank=1)
        # First r rows of beta should form identity
        @test m.beta[1, 1] ≈ 1.0 atol=1e-10

        Y4 = _vecm_dgp2(MersenneTwister(456), 400)
        m2 = estimate_vecm(Y4, 2; rank=2)
        @test m2.beta[1:2, :] ≈ Matrix{Float64}(I, 2, 2) atol=1e-8
    end

    @testset "Johansen result stored" begin
        Y = _vecm_dgp(MersenneTwister(42), 200)
        m = estimate_vecm(Y, 2)
        @test m.johansen_result isa JohansenResult
        @test m.johansen_result.rank >= 0
    end
end

# =============================================================================
# Engle-Granger Estimation
# =============================================================================

@testset "VECM Engle-Granger Estimation" begin

    @testset "Basic bivariate" begin
        Y = _vecm_biv(MersenneTwister(42), 200)
        m = estimate_vecm(Y, 2; method=:engle_granger)
        @test m isa VECMModel{Float64}
        @test m.rank == 1
        @test m.method == :engle_granger
        @test m.johansen_result === nothing
        @test size(m.alpha) == (2, 1)
        @test size(m.beta) == (2, 1)
        @test m.beta[1, 1] ≈ 1.0  # normalized on first variable
    end

    @testset "Multivariate" begin
        d = dgp_vecm(MersenneTwister(42); T=400)
        m = estimate_vecm(d.Y, 2; method=:engle_granger)
        @test m.rank == 1
        @test size(m.alpha) == (3, 1)
        @test size(m.beta) == (3, 1)
        # Both single-equation and system routes estimate the same β (DGP-04
        # #793): probed diff 0.02 on this DGP, bound 0.1 carries a 5x margin.
        joh = estimate_vecm(d.Y, 2; rank=1)
        @test maximum(abs, vec(m.beta) - vec(joh.beta)) < 0.1
    end

    @testset "Rank must be 1" begin
        Y = _vecm_dgp(MersenneTwister(42), 200)
        @test_throws ArgumentError estimate_vecm(Y, 2; method=:engle_granger, rank=2)
    end
end

# =============================================================================
# Rank Zero (No Cointegration)
# =============================================================================

@testset "VECM Rank Zero" begin
    Y = _vecm_dgp(MersenneTwister(42), 200)
    m = estimate_vecm(Y, 2; rank=0)

    @test m.rank == 0
    @test size(m.alpha) == (3, 0)
    @test size(m.beta) == (3, 0)
    @test m.Pi ≈ zeros(3, 3) atol=1e-15
    @test m isa VECMModel{Float64}
    @test isfinite(m.aic)

    # ... and on a genuine rank-0 truth the restriction is correct (DGP-04 #793):
    # a pure random walk has all n roots at unity (plus p − 1 zeros per var).
    Y0 = _vecm_rw(MersenneTwister(12), 400)
    m0 = estimate_vecm(Y0, 2; rank=0)
    @test m0.rank == 0
    v0 = to_var(m0)
    ev0 = sort(abs.(eigvals(companion_matrix(v0.B, nvars(v0), v0.p))); rev=true)
    @test all(abs.(ev0[1:3] .- 1.0) .< 0.05)
end

# =============================================================================
# to_var() Conversion
# =============================================================================

@testset "VECM to VAR Conversion" begin

    @testset "Dimensions" begin
        Y = _vecm_dgp(MersenneTwister(42), 200)
        m = estimate_vecm(Y, 2; rank=1)
        v = to_var(m)

        @test v isa VARModel{Float64}
        @test nvars(v) == 3
        @test v.p == 2
        @test size(v.B) == (1 + 3*2, 3)
        @test size(v.Y) == size(Y)
        @test isfinite(v.aic)
        @test isfinite(v.bic)
    end

    @testset "VAR(1) conversion" begin
        Y = _vecm_dgp(MersenneTwister(42), 200)
        m = estimate_vecm(Y, 1; rank=1)
        v = to_var(m)
        @test v.p == 1
        @test size(v.B) == (1 + 3, 3)
    end

    @testset "Coefficient reconstruction" begin
        # For VAR(2): A1 = Pi + I + Gamma1, A2 = -Gamma1
        Y = _vecm_dgp(MersenneTwister(42), 200)
        m = estimate_vecm(Y, 2; rank=1)
        v = to_var(m)

        n = 3
        In = Matrix{Float64}(I, n, n)
        A1_expected = m.Pi + In + m.Gamma[1]
        A2_expected = -m.Gamma[1]

        A = extract_ar_coefficients(v.B, n, 2)
        @test A[1] ≈ A1_expected atol=1e-10
        @test A[2] ≈ A2_expected atol=1e-10

        # Intercept
        @test v.B[1, :] ≈ m.mu atol=1e-10
    end

    @testset "VAR(3) conversion" begin
        Y = _vecm_dgp(MersenneTwister(42), 200)
        m = estimate_vecm(Y, 3; rank=1)
        v = to_var(m)

        n = 3
        In = Matrix{Float64}(I, n, n)
        A = extract_ar_coefficients(v.B, n, 3)

        # A1 = Pi + I + Gamma1
        @test A[1] ≈ m.Pi + In + m.Gamma[1] atol=1e-10
        # A2 = Gamma2 - Gamma1
        @test A[2] ≈ m.Gamma[2] - m.Gamma[1] atol=1e-10
        # A3 = -Gamma2
        @test A[3] ≈ -m.Gamma[2] atol=1e-10
    end

    @testset "Companion eigenvalues" begin
        Y = _vecm_dgp(MersenneTwister(11), 400)
        m = estimate_vecm(Y, 2; rank=1)
        v = to_var(m)
        F = companion_matrix(v.B, nvars(v), v.p)
        ev = sort(abs.(eigvals(F)); rev=true)
        # Cointegrated rank-1 system: exactly n − r = 2 unit roots (DGP-04
        # #793; probed at exactly 1.0), the rest stationary (probed ≤ 0.49).
        @test all(abs.(ev[1:2] .- 1.0) .< 0.02)
        @test all(ev[3:end] .< 0.9)
    end
end

# =============================================================================
# IRF / FEVD / HD via VECM
# =============================================================================

@testset "VECM Innovation Accounting" begin
    Y = _vecm_dgp(MersenneTwister(42), 200)
    m = estimate_vecm(Y, 2; rank=1)

    @testset "IRF dispatch" begin
        r = irf(m, 10)
        @test r isa ImpulseResponse{Float64}
        @test size(r.values) == (10, 3, 3)

        # With CIs
        r2 = irf(m, 10; ci_type=:bootstrap, reps=50)
        @test r2.ci_type == :bootstrap
        @test size(r2.ci_lower) == (10, 3, 3)
    end

    @testset "FEVD dispatch" begin
        f = fevd(m, 10)
        @test f isa FEVD{Float64}
        @test size(f.proportions) == (3, 3, 10)
        # Proportions sum to 1 at each horizon
        for h in 1:10
            for v in 1:3
                @test sum(f.proportions[v, :, h]) ≈ 1.0 atol=1e-8
            end
        end
    end

    @testset "Historical decomposition dispatch" begin
        T_eff = effective_nobs(to_var(m))
        hd = historical_decomposition(m, T_eff)
        @test hd isa HistoricalDecomposition{Float64}
        @test hd.method == :cholesky
        # Additivity identity (DGP-04 #793): actual = Σ contributions + init.
        @test maximum(abs, hd.actual -
            (dropdims(sum(hd.contributions; dims=3); dims=3) + hd.initial_conditions)) < 1e-8
    end
end

# =============================================================================
# Forecasting
# =============================================================================

@testset "VECM Forecasting" begin
    Y = _vecm_dgp(MersenneTwister(42), 200)
    m = estimate_vecm(Y, 2; rank=1)

    @testset "Point forecast" begin
        fc = forecast(m, 10)
        @test fc isa VECMForecast{Float64}
        @test size(fc.levels) == (10, 3)
        @test size(fc.differences) == (10, 3)
        @test fc.horizon == 10
        @test fc.ci_method == :none
        @test all(isfinite, fc.levels)

        # Differences should be consistent with levels
        expected_diff = diff(vcat(Y[end:end, :], fc.levels), dims=1)
        @test fc.differences ≈ expected_diff atol=1e-10
    end

    @testset "Bootstrap CIs" begin
        # Explicit rng: band construction must be reproducible (DGP-04 #793).
        fc = forecast(m, 5; ci_method=:bootstrap, reps=100, rng=MersenneTwister(3))
        @test fc.ci_method == :bootstrap
        @test size(fc.ci_lower) == (5, 3)
        @test size(fc.ci_upper) == (5, 3)
        # Bands bracket the point forecast everywhere (the old
        # `ci_lower ≤ levels + 1.0` was unfalsifiable).
        @test all(fc.ci_lower .<= fc.levels .<= fc.ci_upper)
    end

    @testset "Bootstrap coverage on known future" begin
        # 80% bands over a 10-draw MC against the DGP's own future (DGP-04
        # #793): probed hit rate 0.75, bound 0.5 carries a wide margin.
        rate = let hits = 0, total = 0
            for seed in 1:10
                d = dgp_vecm(MersenneTwister(100 + seed); T=304)
                mf = estimate_vecm(d.Y[1:300, :], 2; rank=1)
                fc = forecast(mf, 4; ci_method=:bootstrap, reps=50, conf_level=0.8,
                              rng=MersenneTwister(seed))
                hits += sum(fc.ci_lower .<= d.Y[301:304, :] .<= fc.ci_upper)
                total += length(fc.levels)
            end
            hits / total
        end
        @test rate > 0.5
    end

    @testset "Simulation CIs" begin
        fc = forecast(m, 5; ci_method=:simulation, reps=100, rng=MersenneTwister(5))
        @test fc.ci_method == :simulation
        @test all(isfinite, fc.ci_lower)
        @test all(isfinite, fc.ci_upper)
        @test all(fc.ci_lower .<= fc.levels .<= fc.ci_upper)
    end

    @testset "Forecast from rank 0" begin
        m0 = estimate_vecm(Y, 2; rank=0)
        fc = forecast(m0, 5)
        @test all(isfinite, fc.levels)
    end

    @testset "Forecast from VAR(1)" begin
        m1 = estimate_vecm(Y, 1; rank=1)
        fc = forecast(m1, 5)
        @test all(isfinite, fc.levels)
    end

    @testset "VECMForecast has conf_level" begin
        fc = forecast(m, 10; conf_level=0.90)
        @test hasproperty(fc, :conf_level)
        @test fc.conf_level ≈ 0.90
    end
end

# =============================================================================
# Granger Causality
# =============================================================================

@testset "VECM Granger Causality" begin
    Y = _vecm_dgp(MersenneTwister(42), 200)
    m = estimate_vecm(Y, 2; rank=1)

    @testset "Basic test" begin
        # Known causal direction (DGP-04 #793): only equation 1
        # error-corrects, so the long-run channel rejects for effect = 1
        # (probed p ≈ 0.0) and not for effects 2, 3 (probed p ≈ 0.5, 0.66).
        md = estimate_vecm(_vecm_directional(MersenneTwister(14), 400), 2; rank=1)
        @test granger_causality_vecm(md, 2, 1).long_run_pvalue < 0.05
        @test granger_causality_vecm(md, 1, 2).long_run_pvalue > 0.05
        @test granger_causality_vecm(md, 1, 3).long_run_pvalue > 0.05

        g = granger_causality_vecm(m, 1, 2)
        @test g isa VECMGrangerResult{Float64}
        @test g.cause_var == 1
        @test g.effect_var == 2

        # Statistics are non-negative
        @test g.short_run_stat >= 0
        @test g.long_run_stat >= 0
        @test g.strong_stat >= 0

        # P-values in [0, 1]
        @test 0 <= g.short_run_pvalue <= 1
        @test 0 <= g.long_run_pvalue <= 1
        @test 0 <= g.strong_pvalue <= 1

        # Degrees of freedom are positive
        @test g.short_run_df >= 0
        @test g.long_run_df >= 0
        @test g.strong_df >= 0
        @test g.strong_df == g.short_run_df + g.long_run_df
    end

    @testset "All variable pairs" begin
        for i in 1:3, j in 1:3
            if i != j
                g = granger_causality_vecm(m, i, j)
                @test g.cause_var == i
                @test g.effect_var == j
                @test isfinite(g.strong_pvalue)
            end
        end
    end

    @testset "Error for same variable" begin
        @test_throws ArgumentError granger_causality_vecm(m, 1, 1)
    end

    @testset "Error for out of range" begin
        @test_throws ArgumentError granger_causality_vecm(m, 0, 1)
        @test_throws ArgumentError granger_causality_vecm(m, 1, 4)
    end

    @testset "Rank 0 model" begin
        m0 = estimate_vecm(Y, 2; rank=0)
        g = granger_causality_vecm(m0, 1, 2)
        @test g.long_run_stat == 0.0
        @test g.long_run_pvalue == 1.0
        @test g.long_run_df == 0
    end
end

# =============================================================================
# Select Rank
# =============================================================================

@testset "VECM Rank Selection" begin
    # Exact recovery of the rank-1 truth by both criteria (DGP-04 #793; the
    # old `0 ≤ r ≤ 3` passed for a selector returning anything). Rank-0/2
    # truths are covered in "Rank detection" above.
    Y = _vecm_dgp(MersenneTwister(11), 400)

    r_trace = select_vecm_rank(Y, 2; criterion=:trace)
    @test r_trace == 1

    r_max = select_vecm_rank(Y, 2; criterion=:max_eigen)
    @test r_max == 1
end

# =============================================================================
# StatsAPI Interface
# =============================================================================

@testset "VECM StatsAPI" begin
    Y = _vecm_dgp(MersenneTwister(42), 200)
    m = estimate_vecm(Y, 2; rank=1)

    @test coef(m) isa Vector{Float64}
    @test length(coef(m)) > 0
    @test residuals(m) === m.U
    @test nobs(m) == 200
    @test aic(m) ≈ m.aic
    @test bic(m) ≈ m.bic
    @test loglikelihood(m) ≈ m.loglik
    @test islinear(m) == true
    @test dof(m) > 0

    # predict returns in-sample fitted differences
    fitted = predict(m)
    @test size(fitted) == (effective_nobs(m), 3)
end

# =============================================================================
# Edge Cases
# =============================================================================

@testset "VECM Edge Cases" begin

    @testset "Input validation" begin
        Y = _vecm_dgp(MersenneTwister(7), 20)
        @test_throws ArgumentError estimate_vecm(Y, 2; deterministic=:invalid)
        @test_throws ArgumentError estimate_vecm(Y, 2; method=:invalid)
        @test_throws ArgumentError estimate_vecm(Y, 0)
        @test_throws ArgumentError estimate_vecm(Y, 2; rank=-1)
        @test_throws ArgumentError estimate_vecm(Y, 2; rank=4)

        Y_small = _vecm_dgp(MersenneTwister(9), 5)
        @test_throws ArgumentError estimate_vecm(Y_small, 2)
    end

    @testset "Full rank" begin
        Y = _vecm_dgp(MersenneTwister(42), 200)
        m = estimate_vecm(Y, 2; rank=3)
        @test m.rank == 3
        @test size(m.alpha) == (3, 3)
        @test size(m.beta) == (3, 3)
    end

    @testset "Float32 input" begin
        Y = Float32.(_vecm_dgp(MersenneTwister(42), 200))
        m = estimate_vecm(Y, 2; rank=1)
        @test m isa VECMModel{Float32}
    end

    @testset "Integer input" begin
        # ×10 scaling keeps the O(1) equilibrium error well above the rounding
        # grid (DGP-04 #793) — the old 0.1-scale error was destroyed by round.
        Y = round.(Int, _vecm_dgp(MersenneTwister(42), 200) .* 10)
        m = estimate_vecm(Y, 2; rank=1)
        @test m isa VECMModel{Float64}  # promoted via @float_fallback
        @test m.rank == 1
    end

    @testset "Bivariate system" begin
        Y = _vecm_biv(MersenneTwister(42), 200)
        m = estimate_vecm(Y, 2; rank=1)
        @test nvars(m) == 2
        @test m.rank == 1
        v = to_var(m)
        @test nvars(v) == 2
    end
end

# =============================================================================
# Display
# =============================================================================

@testset "VECM Display" begin
    Y = _vecm_dgp(MersenneTwister(42), 200)
    m = estimate_vecm(Y, 2; rank=1)

    @testset "show" begin
        buf = IOBuffer()
        show(buf, m)
        s = String(take!(buf))
        @test occursin("VECM", s)
        @test occursin("Rank", s)
    end

    @testset "report" begin
        # Test show(io, m) directly — report() delegates to show(stdout, m) for VECM
        # but redirect_stdout(devnull) has issues with some backends
        buf = IOBuffer()
        show(buf, m)
        output = String(take!(buf))
        @test occursin("VECM", output)
        @test occursin("Cointegrating", output)
    end

    @testset "VECMForecast show" begin
        fc = forecast(m, 5)
        buf = IOBuffer()
        show(buf, fc)
        s = String(take!(buf))
        @test occursin("VECM Forecast", s)
    end

    @testset "VECMGrangerResult show" begin
        g = granger_causality_vecm(m, 1, 2)
        buf = IOBuffer()
        show(buf, g)
        s = String(take!(buf))
        @test occursin("Granger", s)
    end

    @testset "refs" begin
        buf = IOBuffer()
        refs(buf, m)
        s = String(take!(buf))
        @test occursin("Johansen", s)
    end

    @testset "refs symbol dispatch" begin
        buf = IOBuffer()
        refs(buf, :vecm)
        s = String(take!(buf))
        @test occursin("Johansen", s)

        buf2 = IOBuffer()
        refs(buf2, :engle_granger)
        s2 = String(take!(buf2))
        @test occursin("Engle", s2)
    end
end

# =============================================================================
# Accessor Functions
# =============================================================================

@testset "VECM Accessors" begin
    Y = _vecm_dgp(MersenneTwister(42), 200)
    m = estimate_vecm(Y, 2; rank=1)

    @test nvars(m) == 3
    @test nlags(m) == 2
    @test cointegrating_rank(m) == 1
    @test effective_nobs(m) == size(m.U, 1)
    @test ncoefs(m) > 0
end

@testset "SID-08 long-run on cointegrated systems" begin
    # Cointegrated bivariate truth (DGP-04 #793) instead of a shared-trend
    # construction; stationary VAR truth instead of white noise.
    Yc = dgp_vecm(MersenneTwister(737); alpha=[-0.3, 0.1], beta=[1.0, -1.0],
                  Gamma=Matrix(0.2 * I, 2, 2), T=200).Y
    vecm = estimate_vecm(Yc, 2; rank=1)
    @test_throws IdentificationError identify_long_run(to_var(vecm))
    Ys = dgp_var(MersenneTwister(737); A=[0.5 0.1; 0.05 0.4],
                 B0=Matrix{Float64}(I, 2, 2), T=200).Y
    ms = estimate_var(Ys, 1)
    Q = identify_long_run(ms)
    @test norm(Q' * Q - I(2)) < 1e-8
end

# =============================================================================
# SID-16 structural VECM
# =============================================================================

@testset "SID-16 structural VECM" begin
    if !@isdefined(simulate_common_trend_svec)
        include(joinpath(@__DIR__, "..", "var", "id_dgps.jl"))
    end

    @testset "KPSW recovery, PT, FEVD, long-run IRF" begin
        rng = MersenneTwister(745)
        Tobs = 1000
        Y, _, B0_true, Xi_true = simulate_common_trend_svec(; Tobs=Tobs, rng=rng)
        lr_true = Xi_true * B0_true
        vecm = estimate_vecm(Y, 1; rank=2, deterministic=:none)
        svec = identify_svec(vecm)
        @test svec isa SVECResult
        @test svec.n_permanent == 1
        lr = svec.Xi * svec.B0
        @test maximum(abs, lr[:, 2]) < 1e-8
        @test maximum(abs, lr[:, 3]) < 1e-8
        perm = lr[:, 1]
        truth = lr_true[:, 1]
        perm = sign(dot(perm, truth)) * perm
        @test norm(perm - truth) / norm(truth) < 0.10

        ir_svec = irf(vecm, 200; method=:svec)
        @test ir_svec.values[end, :, 1] ≈ lr[:, 1] atol=1e-2 rtol=0.05

        ir_lr = irf(vecm, 10; method=:long_run)
        @test ir_lr isa ImpulseResponse
        @test ir_lr.values[1, :, :] ≈ irf(vecm, 10; method=:svec).values[1, :, :]

        pt = permanent_transitory(vecm; method=:gonzalo_ng)
        @test pt.permanent + pt.transitory ≈ vecm.Y atol=1e-8
        for j in 1:3
            adf = adf_test(pt.transitory[:, j])
            @test adf.pvalue < 0.05
        end

        fv = fevd(vecm, 200; method=:svec)
        @test all(fv.proportions[:, 2, end] .< 0.15)
        @test all(fv.proportions[:, 3, end] .< 0.15)
    end

    @testset "reject frozen-Q CIs; merge PT zeros; peel IRF kwargs" begin
        rng = MersenneTwister(7451)
        Y, _, _, _ = simulate_common_trend_svec(; Tobs=250, rng=rng)
        vecm = estimate_vecm(Y, 1; rank=2, deterministic=:none)

        @test_throws ArgumentError irf(vecm, 4; method=:svec, ci_type=:bootstrap)
        @test_throws ArgumentError irf(vecm, 4; method=:svec, ci_type=:theoretical)
        @test_throws ArgumentError irf(vecm, 4; method=:long_run, ci_type=:bootstrap)
        @test_throws ArgumentError irf(vecm, 4; method=:long_run, ci_type=:theoretical)
        r = irf(vecm, 4; method=:svec, reps=10, conf_level=0.9)
        @test r.ci_type === :none

        n, n_perm = 3, 1
        lrz = fill(NaN, n, n)
        pat = MacroEconometricModels._svec_resolve_pattern(n, n_perm, nothing, lrz, nothing, Float64)
        @test all(iszero, pat.long_run[:, 2])
        @test all(iszero, pat.long_run[:, 3])

        Bz = fill(NaN, n, n)
        Bz[2, 3] = 0.0
        custom = SVARPattern(Matrix{Float64}(I, n, n), Bz)
        @test custom.long_run === nothing
        svec = identify_svec(vecm; pattern=custom, n_starts=1, rng=MersenneTwister(2))
        lr = svec.Xi * svec.B0
        @test maximum(abs, lr[:, 2]) < 1e-5
        @test maximum(abs, lr[:, 3]) < 1e-5
    end
end
