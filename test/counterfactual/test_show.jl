# CF-21 (#401): report()/refs() integration for the counterfactual types.
using Test
using LinearAlgebra
using Random
using MacroEconometricModels

const MEM = MacroEconometricModels

# Small fixtures of every result type (Float64 and Float32 where cheap).
function _cf21_fixtures(rng)
    H = 6
    Tx = randn(rng, H, 2)
    Tz = randn(rng, H, 2)
    rep3(M) = cat(M, M, M; dims=3)
    ce = PolicyCausalEffects(outcomes=[:infl, :ygap], instruments=[:rate],
                             Theta_x=[Tx, 0.5 .* Tx], Theta_z=[Tz],
                             Theta_x_draws=[rep3(Tx), rep3(0.5 .* Tx)],
                             Theta_z_draws=[rep3(Tz)], source=:var)
    ce_sq = PolicyCausalEffects(outcomes=[:infl, :ygap], instruments=[:rate],
                                Theta_x=[randn(rng, H, H), randn(rng, H, H)],
                                Theta_z=[randn(rng, H, H)], source=:dsge)
    base = MEM.BaselinePath{Float64}([:infl, :ygap], [:rate],
                                     [randn(rng, H), randn(rng, H)], [randn(rng, H)],
                                     nothing, nothing, H, "demand")
    rule = rate_peg_rule(H; outcomes=[:infl, :ygap], instruments=[:rate])
    pc = MEM._suppress_warnings() do
        policy_counterfactual(base, ce, rule)
    end
    loss = policy_loss([:infl, :ygap], H; lambda=[1.0, 0.25])
    po = MEM._suppress_warnings() do
        optimal_policy(base, ce, loss)
    end
    fc = MEM.PolicyForecast{Float64}([:infl, :ygap],
                                     [randn(rng, H), randn(rng, H)],
                                     [randn(rng, H, 8), randn(rng, H, 8)], H, "2021Q2")
    r_opp = MEM._suppress_warnings() do
        estimate_opp(fc, ce, loss; n_sim=60, rng=MersenneTwister(3))
    end
    sq = MEM._suppress_warnings() do
        opp_sequence([fc, fc], ce, loss; n_sim=40, rng=MersenneTwister(4))
    end
    wold = MEM.WoldRepresentation{Float64}(
        cat([Matrix{Float64}(I, 3, 3) .* 0.5^h for h in 0:H-1]...; dims=3) |>
        A -> permutedims(A, (3, 1, 2)), Matrix{Float64}(I, 3, 3),
        ["infl", "ygap", "rate"], nothing)
    cm = MEM._suppress_warnings() do
        counterfactual_moments(wold, ce, rule;
                               outcomes=[:infl => 1, :ygap => 2],
                               instruments=[:rate => 3], warn_invertibility=false)
    end
    sdg = MEM._suppress_warnings() do
        spanning_diagnostic(base, ce, ce_sq, rule; draws=:off)
    end
    return ce, pc, po, r_opp, sq, cm, sdg
end

@testset "report()/refs() integration (CF-21)" begin
    rng = MersenneTwister(20260821)
    ce, pc, po, r_opp, sq, cm, sdg = _cf21_fixtures(rng)

    @testset "honesty strings" begin
        s_pc = sprint(report, pc)
        @test occursin("Rule enforceable within shock span", s_pc)
        @test occursin("rel. residual", s_pc)
        @test occursin("ν", s_pc)

        s_opp = sprint(report, r_opp)
        @test occursin("LOWER levels is the conservative choice", s_opp)
        @test occursin("δ (median)", s_opp)
        @test occursin("δ (plug-in)", s_opp)

        s_cm = sprint(report, cm)
        @test occursin("tail share", s_cm)

        s_sd = sprint(report, sdg)
        @test occursin("Spanning Diagnostic", s_sd)
        @test occursin("Loading inside", s_sd)

        s_sq = sprint(report, sq)
        @test occursin("Share of dates rejecting", s_sq)
        @test occursin("Largest |δ| episodes", s_sq)
    end

    @testset "all types render (Float64)" begin
        for x in (ce, pc, po, r_opp, sq, cm, sdg)
            s = sprint(report, x)
            @test !isempty(s)
            @test report(devnull, x) === nothing     # report returns nothing
        end
        # optimal-policy result surfaces the loss block
        s_po = sprint(report, po)
        @test occursin("Loss baseline", s_po)
        @test occursin("FOC", s_po)
    end

    @testset "history + bank + sufficiency render" begin
        # tiny history fixture
        ch = MEM.CounterfactualHistory{Float64}(["a", "b"], [:x, :z],
                                                randn(rng, 2, 2), randn(rng, 2, 2),
                                                nothing, randn(rng, 2, 2), [0.01, 0.02],
                                                "peg", 6, [0.16, 0.5, 0.84], 0, 0)
        s = sprint(report, ch)
        @test occursin("Historical Counterfactual", s)
        @test occursin("max |gap|", s)

        # bank member fixture (hand-built)
        mb = MEM.ModelBankMember{Float64}("RE", [:kappa], randn(rng, 30, 1),
                                          randn(rng, 30), -12.3,
                                          [ce], 0.31, 3)
        s2 = sprint(report, mb)
        @test occursin("Model Bank Member", s2)
        @test occursin("kappa", s2)
        s3 = sprint(io -> report(io, [mb, mb]))
        @test occursin("Posterior Model Probabilities", s3)

        spec = @dsge begin
            parameters: ρ = 0.8
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + ε[t]
        end
        fs = forecast_sufficiency(solve(spec), [:y]; H=8)
        s4 = sprint(report, fs)
        @test occursin("Forecast Sufficiency", s4)
        @test occursin("sufficient, not necessary", s4)
    end

    @testset "Float32 renders" begin
        H = 4
        ce32 = PolicyCausalEffects(outcomes=[:x], Theta_x=[ones(Float32, H, 2)])
        @test !isempty(sprint(report, ce32))
    end

    @testset "refs for every type in all formats" begin
        spec = @dsge begin
            parameters: ρ = 0.8
            endogenous: y
            exogenous: ε
            y[t] = ρ * y[t-1] + ε[t]
        end
        fs = forecast_sufficiency(solve(spec), [:y]; H=6)
        ch = MEM.CounterfactualHistory{Float64}(["a"], [:x], ones(1, 1), ones(1, 1),
                                                nothing, ones(1, 1), [0.0],
                                                "peg", 4, [0.5], 0, 0)
        mb = MEM.ModelBankMember{Float64}("RE", [:k], ones(20, 1), ones(20), -1.0,
                                          [ce], 0.3, 2)
        objs = Any[ce, pc, r_opp, sq, cm, sdg, fs, ch, mb]
        for x in objs, fmt in (:text, :latex, :bibtex, :html)
            s = sprint((io, o) -> refs(io, o; format=fmt), x)
            @test !isempty(s)
        end
        # key spot checks
        @test occursin("Barnichon", sprint(refs, r_opp))
        @test occursin("McKay", sprint(refs, pc))
        @test occursin("Caravello", sprint(refs, sdg))
        @test occursin("Geweke", sprint(refs, mb))
    end
end
