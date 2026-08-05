# CF-22 (#402): plot_result dispatches for the counterfactual result types.
# Structural assertions run on EXTRACTED JSON, never raw p.html (plotrule).
using Test
using LinearAlgebra
using Random
using MacroEconometricModels

include(joinpath(@__DIR__, "..", "plotting", "plot_test_helpers.jl"))

const MEM = MacroEconometricModels

function _cf22_pc(rng; spanned::Bool)
    H = 8
    if spanned
        Tx = randn(rng, H, H)
        Tz = randn(rng, H, H)
        ce = PolicyCausalEffects(outcomes=[:infl], instruments=[:rate],
                                 Theta_x=[Tx], Theta_z=[Tz])
    else
        Tx = reshape([-(0.3^t) for t in 1:H], H, 1)
        Tz = reshape([(0.3^t) for t in 1:H], H, 1)
        ce = PolicyCausalEffects(outcomes=[:infl], instruments=[:rate],
                                 Theta_x=[Tx], Theta_z=[Tz])
    end
    base = MEM.BaselinePath{Float64}([:infl], [:rate],
                                     [[0.95^t for t in 1:H]], [zeros(H)],
                                     nothing, nothing, H, "d")
    rule = inflation_target_rule(H; outcomes=[:infl], instruments=[:rate])
    MEM._suppress_warnings() do
        policy_counterfactual(base, ce, rule)
    end
end

@testset "Counterfactual plotting (CF-22)" begin
    rng = MersenneTwister(20260822)

    @testset "PolicyCounterfactual paths + error panel" begin
        pc_ok = _cf22_pc(rng; spanned=true)
        p = plot_result(pc_ok)
        @test p isa PlotOutput
        check_plot(p)
        @test !occursin("Implementation error", p.html)   # spanned: no error panel
        @test occursin("Baseline", p.html)
        @test occursin("Counterfactual", p.html)

        pc_bad = _cf22_pc(rng; spanned=false)
        @test pc_bad.rel_residual > 0.05
        p2 = plot_result(pc_bad)
        @test occursin("Implementation error", p2.html)   # auto honesty panel

        # vars selection honored
        p3 = plot_result(pc_ok; vars=[:infl])
        @test count("panel-title", p3.html) < count("panel-title", p.html)
        @test_throws ArgumentError plot_result(pc_ok; vars=[:nope])
    end

    @testset "OPPResult delta + paths" begin
        H = 6
        Tx = randn(rng, H, 2)
        noises = 0.1 .* randn(MersenneTwister(1), 30)
        Dx = cat((Tx .* (1 + e) for e in noises)...; dims=3)
        ce = PolicyCausalEffects(outcomes=[:u], Theta_x=[Tx], Theta_x_draws=[Dx])
        fc = MEM.PolicyForecast{Float64}([:u], [randn(rng, H)], nothing, H, "t")
        loss = policy_loss([:u], H; lambda=[1.0])
        r_pt = opp(fc, ce, loss)
        p = plot_result(r_pt)                              # plug-in: bar form
        @test p isa PlotOutput
        check_plot(p)
        r = estimate_opp(fc, ce, loss; n_sim=200, rng=MersenneTwister(2))
        p2 = plot_result(r)                                # bands: whisker form
        @test occursin("LOWER levels", p2.html)            # polarity in the panel title
        p3 = plot_result(r; view=:paths)
        @test occursin("Announced", p3.html)
        @test_throws ArgumentError plot_result(r; view=:bogus)
    end

    @testset "OPPSequence fan nests correctly" begin
        H = 6
        Tx = randn(rng, H, 1)
        noises = 0.1 .* randn(MersenneTwister(3), 30)
        Dx = cat((Tx .* (1 + e) for e in noises)...; dims=3)
        ce = PolicyCausalEffects(outcomes=[:u], Theta_x=[Tx], Theta_x_draws=[Dx])
        loss = policy_loss([:u], H; lambda=[1.0])
        fcs = [MEM.PolicyForecast{Float64}([:u], [randn(rng, H)], nothing, H, "d$q")
               for q in 1:5]
        sq = opp_sequence(fcs, ce, loss; n_sim=200, rng=MersenneTwister(4),
                          dates=["d$q" for q in 1:5])
        p = plot_result(sq; view=:fan)
        @test p isa PlotOutput
        check_plot(p)
        # three nested band pairs from levels 60/75/90, ordered outer→inner
        @test occursin("5–95%", p.html)
        @test occursin("12–88%", p.html)                   # (1±0.75)/2 → 12.5/87.5 → rounded
        @test occursin("20–80%", p.html)
        p2 = plot_result(sq; view=:decomposition)
        @test occursin("time-consistent", p2.html)
        # date labels appear as tick labels (line renderer carries the tick map;
        # the fan renderer has no date-tick option — numeric date index there)
        @test occursin("\"label\":\"d1\"", p2.html)
        @test_throws ArgumentError plot_result(sq; shock=9)
    end

    @testset "CounterfactualMoments sd + corr views" begin
        H = 20
        w = MEM.WoldRepresentation{Float64}(
            permutedims(cat(([0.5^h * Matrix{Float64}(I, 2, 2) for h in 0:H-1])...;
                            dims=3), (3, 1, 2)),
            Matrix{Float64}(I, 2, 2), ["u", "rate"], nothing)
        ce = PolicyCausalEffects(outcomes=[:u], instruments=[:rate],
                                 Theta_x=[randn(rng, H, 2)], Theta_z=[randn(rng, H, 2)])
        peg = rate_peg_rule(H; outcomes=[:u], instruments=[:rate])
        cm = MEM._suppress_warnings() do
            counterfactual_moments(w, ce, peg; outcomes=[:u => 1],
                                   instruments=[:rate => 2],
                                   warn_invertibility=false)
        end
        p = plot_result(cm)
        @test p isa PlotOutput
        check_plot(p)
        @test occursin("Std. dev.", p.html)
        p2 = plot_result(cm; view=:corr)
        @test occursin("Baseline correlations", p2.html)
        @test occursin("Counterfactual correlations", p2.html)
    end

    @testset "SpanningDiagnostic + History + Sufficiency" begin
        H = 8
        Txf = randn(rng, H, H)
        Tzf = randn(rng, H, H)
        ce_full = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                      Theta_x=[Txf], Theta_z=[Tzf])
        ce_emp = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                     Theta_x=[Txf[:, 1:2]], Theta_z=[Tzf[:, 1:2]])
        base = MEM.BaselinePath{Float64}([:x], [:z], [randn(rng, H)], [zeros(H)],
                                         nothing, nothing, H, "b")
        rule = inflation_target_rule(H; pi_var=:x, outcomes=[:x], instruments=[:z])
        sdg = MEM._suppress_warnings() do
            spanning_diagnostic(base, ce_emp, ce_full, rule)
        end
        p = plot_result(sdg)
        @test occursin("Empirics only", p.html)
        @test occursin("Model extrapolated", p.html)
        @test occursin("loading inside span", p.html)
        check_plot(p)

        ch = MEM.CounterfactualHistory{Float64}(["1990Q1", "1990Q2", "1990Q3"],
                                                [:x, :z], randn(rng, 3, 2),
                                                randn(rng, 3, 2), nothing,
                                                randn(rng, 1, 3), [0.0, 0.0, 0.0],
                                                "peg", 8, [0.5], 0, 0)
        p2 = plot_result(ch)
        @test occursin("Realized", p2.html)
        @test occursin("\"label\":\"1990Q1\"", p2.html)    # real date labels, no fake axis
        check_plot(p2)

        spec = @dsge begin
            parameters: ρ = 0.8
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + ε[t]
        end
        fs = forecast_sufficiency(solve(spec), [:y]; H=10)
        p3 = plot_result(fs)
        @test occursin("FEV ratio", p3.html)
        check_plot(p3)
    end

    @testset "self-containment, ids, save round-trip" begin
        pc = _cf22_pc(rng; spanned=true)
        p1 = plot_result(pc)
        p2 = plot_result(pc)
        assert_self_contained(p1.html)
        # unique ids across two figures on one page
        id1 = match(r"id=\"(cfpath_\d+)\"", p1.html)
        id2 = match(r"id=\"(cfpath_\d+)\"", p2.html)
        @test id1 !== nothing && id2 !== nothing
        @test id1.captures[1] != id2.captures[1]
        mktempdir() do dir
            f = joinpath(dir, "cf.html")
            plot_result(pc; save_path=f)
            @test isfile(f)
            @test occursin("cfpath", read(f, String))
        end
    end
end
