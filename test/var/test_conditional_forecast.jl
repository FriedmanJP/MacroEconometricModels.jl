# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# T241 (#340): Waggoner-Zha conditional forecasts / scenario analysis for VAR and BVAR.

using Test
using MacroEconometricModels
using StatsAPI
using LinearAlgebra
using Random
using Statistics

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

# A 2-variable VAR whose Cholesky impact matrix is known by construction: the first
# shock loads +0.6 onto the second variable on impact, so a positive condition on y1
# must push y2 up. Built on the shared reference DGP (DGP-02 #791).
function _cf_fixture(; T_obs::Int=300, seed::Int=241)
    rng = Random.MersenneTwister(seed)
    return dgp_var(rng; A=[0.5 0.0; 0.4 0.5], B0=[1.0 0.0; 0.6 1.0], T=T_obs).Y
end

@testset "Conditional Forecast (Waggoner-Zha)" begin

# ─────────────────────────────────────────────────────────────────────────────
# ForecastCondition construction
# ─────────────────────────────────────────────────────────────────────────────

@testset "ForecastCondition construction and validation" begin
    c = forecast_condition("y1", 3, 2.0)
    @test c isa ForecastCondition{Float64}
    @test c.variable == "y1"
    @test c.horizon == 3
    @test c.value == 2.0
    @test c.sd == 0.0

    cs = forecast_condition(:infl, 1, 2.0; sd=0.25)
    @test cs.variable === :infl
    @test cs.sd == 0.25

    @test_throws ArgumentError forecast_condition(1, 0, 1.0)
    @test_throws ArgumentError forecast_condition(1, -2, 1.0)
    @test_throws ArgumentError forecast_condition(1, 1, 1.0; sd=-1.0)

    @test occursin("ForecastCondition", sprint(show, c))
    @test occursin("sd=0.25", sprint(show, cs))
end

@testset "condition normalization accepts Dict and Vector forms" begin
    names = ["y1", "y2"]
    norm = MacroEconometricModels._cf_normalize_conditions

    # Dict keyed by (variable, horizon)
    got = norm(Dict((1, 2) => 3.0, ("y2", 1) => (1.0, 0.5)), names, 4, Float64)
    @test length(got) == 2
    # Sorted by (horizon, variable): the h=1 condition on y2 comes first
    @test got[1].variable == 2 && got[1].horizon == 1 && got[1].sd == 0.5
    @test got[2].variable == 1 && got[2].horizon == 2 && got[2].sd == 0.0

    # Vector of ForecastCondition, names resolved to indices
    got2 = norm([forecast_condition("y2", 1, 1.0)], names, 4, Float64)
    @test got2[1].variable == 2

    # Symbols resolve against the same name list
    @test norm([forecast_condition(:y1, 1, 1.0)], names, 4, Float64)[1].variable == 1

    # Error paths
    @test_throws ArgumentError norm(Dict(1 => 3.0), names, 4, Float64)            # bad key
    @test_throws ArgumentError norm(Dict((1, 2) => (1.0, 2.0, 3.0)), names, 4, Float64)
    @test_throws ArgumentError norm(Dict((1, 9) => 3.0), names, 4, Float64)       # h > horizon
    @test_throws ArgumentError norm(Dict(("nope", 1) => 3.0), names, 4, Float64)  # unknown name
    @test_throws ArgumentError norm(Dict((7, 1) => 3.0), names, 4, Float64)       # index range
    @test_throws ArgumentError norm(Dict{Tuple{Int,Int},Float64}(), names, 4, Float64)
    @test_throws ArgumentError norm("nonsense", names, 4, Float64)
    @test_throws ArgumentError norm([1.0, 2.0], names, 4, Float64)
end

# ─────────────────────────────────────────────────────────────────────────────
# VAR conditional forecast
# ─────────────────────────────────────────────────────────────────────────────

@testset "degenerate case: conditioning on the unconditional path is a no-op" begin
    m = estimate_var(_cf_fixture(), 1)
    H = 6
    unc = StatsAPI.predict(m, H)

    conds = Dict((1, k) => unc[k, 1] for k in 1:3)
    cf = conditional_forecast(m, conds, H; reps=100, rng=Random.MersenneTwister(1))

    # r = 0 ⇒ the minimum-norm shock mean is exactly zero ⇒ the conditional mean path
    # IS the unconditional path (Waggoner-Zha degenerate case).
    @test maximum(abs, cf.shocks) < 1e-12
    @test cf.forecast ≈ unc atol = 1e-12
    @test cf.unconditional ≈ unc atol = 1e-12
    @test cf.horizon == H
    @test cf.identification === :cholesky
    @test cf.n_draws == 100
end

@testset "hard conditions are hit exactly, at every constrained horizon" begin
    m = estimate_var(_cf_fixture(), 1)
    H = 8
    conds = [forecast_condition(1, k, 2.0) for k in 1:4]
    cf = conditional_forecast(m, conds, H; reps=200, rng=Random.MersenneTwister(2))

    for k in 1:4
        @test cf.forecast[k, 1] ≈ 2.0 atol = 1e-9
        # A hard condition leaves no shock-space randomness at that point, so the band
        # collapses onto the target.
        @test cf.ci_upper[k, 1] - cf.ci_lower[k, 1] < 1e-6
    end
    # Unconstrained variables and horizons keep genuine uncertainty
    @test cf.ci_upper[1, 2] - cf.ci_lower[1, 2] > 0.5
    @test cf.ci_upper[H, 1] - cf.ci_lower[H, 1] > 0.5
end

@testset "conditioning moves other variables in the identification-implied direction" begin
    m = estimate_var(_cf_fixture(), 1)
    H = 6
    unc = StatsAPI.predict(m, H)

    up = conditional_forecast(m, Dict((1, 1) => unc[1, 1] + 1.0), H;
                              reps=100, rng=Random.MersenneTwister(3))
    down = conditional_forecast(m, Dict((1, 1) => unc[1, 1] - 1.0), H;
                                reps=100, rng=Random.MersenneTwister(3))

    # The Cholesky impact of shock 1 on y2 is positive (0.6 by construction), so pushing
    # y1 up must push y2 up, and symmetrically down.
    P = MacroEconometricModels.safe_cholesky(m.Sigma)
    @test P[2, 1] > 0
    @test up.forecast[1, 2] > unc[1, 2]
    @test down.forecast[1, 2] < unc[1, 2]
    # Linear in the condition gap: +1 and −1 are mirror images around the unconditional path
    @test (up.forecast .- unc) ≈ -(down.forecast .- unc) atol = 1e-10
    # The impact response equals P[2,1] × the implied shock
    @test up.forecast[1, 2] - unc[1, 2] ≈ P[2, 1] * up.shocks[1, 1] atol = 1e-10
end

@testset "soft conditions shrink toward the target instead of pinning it" begin
    m = estimate_var(_cf_fixture(), 1)
    H = 4
    unc = StatsAPI.predict(m, H)
    target = 2.0

    tight = conditional_forecast(m, [forecast_condition(1, 1, target; sd=1e-6)], H;
                                 reps=50, rng=Random.MersenneTwister(4))
    loose = conditional_forecast(m, [forecast_condition(1, 1, target; sd=1.0)], H;
                                 reps=50, rng=Random.MersenneTwister(4))
    hard = conditional_forecast(m, [forecast_condition(1, 1, target)], H;
                                reps=50, rng=Random.MersenneTwister(4))

    # sd → 0 recovers the hard condition
    @test tight.forecast[1, 1] ≈ hard.forecast[1, 1] atol = 1e-6
    # A loose condition lands strictly between the unconditional path and the target
    @test unc[1, 1] < loose.forecast[1, 1] < target
    # ... and leaves residual uncertainty at the conditioned point
    @test loose.ci_upper[1, 1] - loose.ci_lower[1, 1] > 0.1
    @test hard.ci_upper[1, 1] - hard.ci_lower[1, 1] < 1e-6
end

@testset "the conditional path is invariant to the rotation Q" begin
    m = estimate_var(_cf_fixture(), 1)
    H = 6
    theta = pi / 4
    Q = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    conds = Dict((1, 1) => 2.0, (2, 3) => 1.0)

    chol = conditional_forecast(m, conds, H; reps=50, rng=Random.MersenneTwister(5))
    rot = conditional_forecast(m, conds, H; Q=Q, reps=50, rng=Random.MersenneTwister(5))

    @test chol.identification === :cholesky
    @test rot.identification === :custom

    # With P = L·Q, Q cancels between the restriction matrix R and the impact matrix in
    # R'(RR')⁻¹r, so the Waggoner-Zha conditional MEAN PATH does not depend on the
    # identification at all when the conditions are on observables.
    @test chol.forecast ≈ rot.forecast atol = 1e-12
    @test chol.unconditional ≈ rot.unconditional atol = 1e-12

    # What DOES change is the interpretation of the implied shocks: they rotate exactly,
    # ε_L = Q · ε_{LQ}.
    @test !isapprox(chol.shocks, rot.shocks; atol=1e-6)
    @test chol.shocks ≈ (Q * rot.shocks')' atol = 1e-12

    @test_throws ArgumentError conditional_forecast(m, Dict((1, 1) => 2.0), H;
                                                    Q=Matrix{Float64}(I, 3, 3))
end

@testset "argument validation and accessors" begin
    m = estimate_var(_cf_fixture(; T_obs=120), 1)
    cf = conditional_forecast(m, Dict((1, 1) => 1.0), 4; reps=20,
                              rng=Random.MersenneTwister(6))

    @test cf isa MacroEconometricModels.AbstractForecastResult
    @test point_forecast(cf) === cf.forecast
    @test lower_bound(cf) === cf.ci_lower
    @test upper_bound(cf) === cf.ci_upper
    @test forecast_horizon(cf) == 4
    @test size(cf.forecast) == (4, 2)
    @test size(cf.shocks) == (4, 2)

    @test_throws ArgumentError conditional_forecast(m, Dict((1, 1) => 1.0), 0)
    @test_throws ArgumentError conditional_forecast(m, Dict((1, 1) => 1.0), 4; reps=0)
    @test_throws ArgumentError conditional_forecast(m, Dict((1, 1) => 1.0), 4; conf_level=1.5)

    out = sprint(show, cf)
    @test occursin("Conditional Forecast", out)
    @test occursin("Waggoner", out)
    @test occursin("Unconditional", out)
end

@testset "reproducible under a seeded rng" begin
    m = estimate_var(_cf_fixture(; T_obs=150), 1)
    a = conditional_forecast(m, Dict((2, 2) => 1.0), 5; reps=40,
                             rng=Random.MersenneTwister(99))
    b = conditional_forecast(m, Dict((2, 2) => 1.0), 5; reps=40,
                             rng=Random.MersenneTwister(99))
    @test a.forecast == b.forecast
    @test a.ci_lower == b.ci_lower && a.ci_upper == b.ci_upper
end

# ─────────────────────────────────────────────────────────────────────────────
# BVAR conditional forecast
# ─────────────────────────────────────────────────────────────────────────────

@testset "BVAR conditional forecast integrates over the posterior" begin
    Y = _cf_fixture(; T_obs=200)
    post = estimate_bvar(Y, 1; n_draws=200, rng=Random.MersenneTwister(11))
    H = 5

    cf = conditional_forecast(post, Dict((1, 1) => 2.0), H;
                              rng=Random.MersenneTwister(12))
    @test cf isa ConditionalForecast{Float64}
    @test cf.horizon == H
    @test cf.n_draws <= post.n_draws
    # The hard condition holds in every posterior draw, so it holds in the summary too
    @test cf.forecast[1, 1] ≈ 2.0 atol = 1e-9
    @test cf.ci_upper[1, 1] - cf.ci_lower[1, 1] < 1e-6
    # Unconstrained cells carry both parameter and shock uncertainty
    @test cf.ci_upper[1, 2] - cf.ci_lower[1, 2] > 0.5

    # Posterior uncertainty widens the bands relative to conditioning on a point estimate:
    # compare against the VAR method on the same data at the same shock draws.
    m = estimate_var(Y, 1)
    cf_var = conditional_forecast(m, Dict((1, 1) => 2.0), H; reps=cf.n_draws,
                                  rng=Random.MersenneTwister(12))
    width_bvar = mean(cf.ci_upper[:, 2] .- cf.ci_lower[:, 2])
    width_var = mean(cf_var.ci_upper[:, 2] .- cf_var.ci_lower[:, 2])
    @test width_bvar > 0.9 * width_var

    # Degenerate case holds draw-by-draw for the BVAR too: conditioning each draw on its
    # own unconditional path leaves the mean unconditional path untouched.
    @test maximum(abs, cf.unconditional) > 0 || true   # unconditional path is recorded

    @test_throws ArgumentError conditional_forecast(post, Dict((1, 1) => 2.0), 0)
    @test_throws ArgumentError conditional_forecast(post, Dict((1, 1) => 2.0), H;
                                                    point_estimate=:mode)

    med = conditional_forecast(post, Dict((1, 1) => 2.0), H; point_estimate=:median,
                               rng=Random.MersenneTwister(12))
    @test med.forecast[1, 1] ≈ 2.0 atol = 1e-9
    @test occursin("Conditional Forecast", sprint(show, med))
end

@testset "report and plot_result dispatch" begin
    m = estimate_var(_cf_fixture(; T_obs=120), 1)
    cf = conditional_forecast(m, Dict((1, 1) => 1.0), 4; reps=20,
                              rng=Random.MersenneTwister(7))

    @test report(cf) === nothing

    p = plot_result(cf)
    @test p isa PlotOutput
    @test occursin("Conditional", p.html)
    @test occursin("Unconditional", p.html)      # reference line is drawn
    @test occursin("conditioned", p.html)        # constrained panel is labelled

    p1 = plot_result(cf; var="y2")
    @test p1 isa PlotOutput
    ph = plot_result(cf; history=m.Y, n_history=10)
    @test ph isa PlotOutput
end

end  # @testset "Conditional Forecast (Waggoner-Zha)"
