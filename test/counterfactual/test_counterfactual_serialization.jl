# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

const _RSER10 = ("BaselinePath", "CounterfactualHistory", "CounterfactualMoments",
                 "ForecastSufficiency", "FunctionConstraint", "ModelBankMember",
                 "OPPResult", "OPPSequence", "PathFloorConstraint",
                 "PolicyCausalEffects", "PolicyCounterfactual", "PolicyForecast",
                 "PolicyLoss", "PolicyRule", "SpanningDiagnostic",
                 "WoldRepresentation")

# Named Main callback for FunctionConstraint (anonymous g must error at save).
rser10_g(δ, paths) = [1.0]

function _rser10_ce(; H=6, n_s=2, draws=false, rng=MersenneTwister(783))
    Tx = [randn(rng, H, n_s), 0.5 .* randn(rng, H, n_s)]
    Tz = [randn(rng, H, n_s)]
    kw = draws ? (; Theta_x_draws=[cat((Tx[1] for _ in 1:4)...; dims=3),
                                   cat((Tx[2] for _ in 1:4)...; dims=3)],
                    Theta_z_draws=[cat((Tz[1] for _ in 1:4)...; dims=3)]) : (;)
    PolicyCausalEffects(outcomes=[:u, :infl], instruments=[:rate],
                        Theta_x=Tx, Theta_z=Tz, source=:var; kw...)
end

function _rser10_fixtures()
    rng = MersenneTwister(783)
    H, n_s = 6, 2
    ce = _rser10_ce(; H=H, n_s=n_s, rng=rng)
    ce_d = _rser10_ce(; H=H, n_s=n_s, draws=true, rng=MersenneTwister(7831))
    rule = PolicyRule(outcomes=[:u, :infl], instruments=[:rate],
                      A_x=[Matrix{Float64}(I, H, H), zeros(H, H)],
                      A_z=[Matrix{Float64}(I, H, H)], name="taylor")
    loss = policy_loss([:u, :infl], H; lambda=[2.0, 1.0], beta=0.97)
    loss_z = policy_loss([:u], H; lambda=[1.0], instruments=[:rate],
                         W_z=[Matrix{Float64}(I, H, H)])
    base = BaselinePath{Float64}([:u, :infl], [:rate],
                                 [randn(rng, H), randn(rng, H)], [zeros(H)],
                                 nothing, nothing, H, "demand")
    base_d = BaselinePath{Float64}([:u], [:rate], [ones(H)], [zeros(H)],
                                   [randn(rng, H, 3)], [zeros(H, 3)], H, "demand")
    fc = PolicyForecast{Float64}([:u, :infl], [randn(rng, H), randn(rng, H)],
                                 nothing, H, "2021Q2")
    fc_d = PolicyForecast{Float64}([:u], [randn(rng, H)], [randn(rng, H, 3)],
                                   H, "2021Q2")
    Y = randn(MersenneTwister(7832), 80, 2)
    mvar = estimate_var(Y, 1)
    wold = wold_representation(mvar; H=H)
    wold_d = WoldRepresentation{Float64}(wold.Theta, wold.Sigma_u, wold.varnames,
                                         cat((wold.Theta for _ in 1:3)...; dims=4))
    pc = _MEM._suppress_warnings() do
        Tx = randn(rng, H, H)
        Tz = randn(rng, H, H)
        ce_sq = PolicyCausalEffects(outcomes=[:infl], instruments=[:rate],
                                    Theta_x=[Tx], Theta_z=[Tz])
        b = BaselinePath{Float64}([:infl], [:rate],
                                  [[0.9^t for t in 1:H]], [zeros(H)],
                                  nothing, nothing, H, "d")
        policy_counterfactual(b, ce_sq,
                              inflation_target_rule(H; outcomes=[:infl],
                                                    instruments=[:rate]))
    end
    cm = _MEM._suppress_warnings() do
        Theta = zeros(H, 3, 3)
        for j in 1:3, v in 1:2, h in 1:H
            Theta[h, v, j] = (v == j ? 1.0 : 0.2) * 0.6^(h - 1)
        end
        w = WoldRepresentation{Float64}(Theta, Matrix{Float64}(I, 3, 3),
                                        ["x1", "x2", "z"], nothing)
        ce_m = PolicyCausalEffects(outcomes=[:x1, :x2], instruments=[:z],
                                   Theta_x=[randn(rng, H, 2), randn(rng, H, 2)],
                                   Theta_z=[randn(rng, H, 2)])
        peg = rate_peg_rule(H; outcomes=[:x1, :x2], instruments=[:z])
        counterfactual_moments(w, ce_m, peg;
                               outcomes=[:x1 => 1, :x2 => 2],
                               instruments=[:z => 3],
                               warn_invertibility=false)
    end
    cm_band = CounterfactualMoments{Float64}(
        cm.varnames, cm.Sigma_base, cm.Sigma_cf, cm.sd_base, cm.sd_cf,
        cm.corr_base, cm.corr_cf, cm.sd_cf_bands, cm.Theta_cf, cm.policy_name,
        cm.H, cm.tail_share, (0.0, π))
    opp_pt = opp(fc, ce, loss)
    opp_sim = _MEM._suppress_warnings() do
        estimate_opp(PolicyForecast{Float64}([:u], [randn(rng, H)], nothing, H, "t"),
                     PolicyCausalEffects(outcomes=[:u], Theta_x=[ce.Theta_x[1]],
                                         Theta_x_draws=[ce_d.Theta_x_draws[1]]),
                     policy_loss([:u], H; lambda=[1.0]);
                     n_sim=40, rng=MersenneTwister(7833))
    end
    fcs = [PolicyForecast{Float64}([:u], [randn(rng, H)], nothing, H, "d$q")
           for q in 1:3]
    sq = opp_sequence(fcs,
                      PolicyCausalEffects(outcomes=[:u], Theta_x=[ce.Theta_x[1]]),
                      policy_loss([:u], H; lambda=[1.0]))
    menus = [PolicyCausalEffects(outcomes=[:pi], Theta_x=[ones(4, 4)], source=:dsge),
             PolicyCausalEffects(outcomes=[:pi], Theta_x=[0.9 .* ones(4, 4)], source=:dsge)]
    mb = ModelBankMember{Float64}("RE", [:kappa], reshape([0.10, 0.12], 2, 1),
                                  [-1.0, -1.1], -10.0, menus, 0.35, 3)
    ch = CounterfactualHistory{Float64}(
        ["t1", "t2"], [:pi, :y, :rate], randn(rng, 2, 3), randn(rng, 2, 3),
        nothing, randn(rng, 2, 2), [0.01, 0.02], "taylor", 8,
        [0.16, 0.5, 0.84], 0, 0)
    sd = let
        Tx = randn(rng, H, H)
        Tz = randn(rng, H, H)
        ce_full = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                      Theta_x=[Tx], Theta_z=[Tz])
        ce_emp = PolicyCausalEffects(outcomes=[:x], instruments=[:z],
                                     Theta_x=[Tx[:, 1:2]], Theta_z=[Tz[:, 1:2]])
        b = BaselinePath{Float64}([:x], [:z], [randn(rng, H)], [zeros(H)],
                                  nothing, nothing, H, "eng")
        r = inflation_target_rule(H; pi_var=:x, outcomes=[:x], instruments=[:z])
        _MEM._suppress_warnings() do
            spanning_diagnostic(b, ce_emp, ce_full, r)
        end
    end
    spec = @dsge begin
        parameters: ρ = 0.8
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + ε[t]
    end
    fs = forecast_sufficiency(solve(spec), [:y]; H=12)
    floor_c = zlb_constraint(; floor=0.0, instrument=:rate, horizons=1:4)
    fcon = FunctionConstraint(rser10_g, 1, "pledge")

    return Pair{String,Any}[
        "BaselinePath" => base,
        "CounterfactualHistory" => ch,
        "CounterfactualMoments" => cm,
        "ForecastSufficiency" => fs,
        "FunctionConstraint" => fcon,
        "ModelBankMember" => mb,
        "OPPResult" => opp_pt,
        "OPPSequence" => sq,
        "PathFloorConstraint" => floor_c,
        "PolicyCausalEffects" => ce_d,
        "PolicyCounterfactual" => pc,
        "PolicyForecast" => fc_d,
        "PolicyLoss" => loss_z,
        "PolicyRule" => rule,
        "SpanningDiagnostic" => sd,
        "WoldRepresentation" => wold_d,
    ], (; ce, ce_d, loss, fc, opp_pt, opp_sim, mb, menus, base_d, cm_band, floor_c)
end

@testset "RSER-10 counterfactual/OPP serialization (#783)" begin
    fixtures, extra = _rser10_fixtures()
    byname = Dict{String,Any}(fixtures)

    @testset "registry" begin
        @test length(_RSER10) == 16
        for name in _RSER10
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
        @test !any(v == "pending RSER-10" for v in values(_MEM._SERIALIZATION_EXCLUDED))
        @test Set(first.(fixtures)) == Set(_RSER10)
    end

    @testset "generic reconstruct + roundtrip + report ($name)" for (name, r) in fixtures
        Tw = Base.typename(typeof(r)).wrapper
        @test string(nameof(Tw)) == name
        @test _from_serializable_is_generic(Tw)
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
    end

    @testset "PathFloorConstraint UnitRange" begin
        c = extra.floor_c
        @test c.horizons isa UnitRange{Int}
        payload = _MEM._to_serializable(c)
        @test payload["horizons"] isa AbstractDict
        @test payload["horizons"]["__unitrange__"] === true
        @test payload["horizons"]["start"] == 1
        @test payload["horizons"]["stop"] == 4
        c2 = _assert_roundtrip(c)
        @test c2.horizons isa UnitRange{Int}
        @test c2.horizons == 1:4
        zlb = zlb_constraint()
        z2 = _assert_roundtrip(zlb)
        @test z2.horizons isa UnitRange{Int}
        @test z2.horizons == 1:typemax(Int)
        @test z2.instrument === :rate
    end

    @testset "FunctionConstraint named-function codec" begin
        fcon = byname["FunctionConstraint"]
        payload = _MEM._to_serializable(fcon)
        @test payload["g"] isa AbstractDict
        @test payload["g"]["__function__"] == "rser10_g"
        @test payload["g"]["module"] == "Main"
        f2 = _assert_roundtrip(fcon)
        @test f2.g === rser10_g
        @test f2.n_out == 1
        @test f2.name == "pledge"
        _MEM._assert_plain_payload(_MEM._build_container(fcon))

        anon = FunctionConstraint((d, p) -> [1.0], 1, "pledge")
        msg = "FunctionConstraint :pledge wraps an anonymous function; define it as a named function to make the OPP result saveable"
        @test_throws SerializationError(msg) _MEM._build_container(anon)
        @test_throws SerializationError(msg) save_model(anon, joinpath(mktempdir(), "anon.jld2"))
    end

    @testset "PolicyCausalEffects draws + nested Vector{Matrix}" begin
        ce = extra.ce_d
        @test ce.Theta_x_draws isa Vector{<:Array{<:AbstractFloat,3}}
        payload = _MEM._to_serializable(ce)
        @test payload["Theta_x"][1] isa AbstractMatrix
        ce2 = _assert_roundtrip(ce)
        @test ce2.Theta_x_draws isa Vector{<:Array{<:AbstractFloat,3}}
        @test size(ce2.Theta_x_draws[1]) == size(ce.Theta_x_draws[1])
        @test ce2.source === :var
        @test is_square(ce2) == is_square(ce)
    end

    @testset "OPPResult bands/reject Dict" begin
        r = extra.opp_sim
        @test r.bands isa AbstractDict
        @test r.reject isa AbstractDict
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
        @test r2.bands isa AbstractDict
        @test Set(keys(r2.bands)) == Set(keys(r.bands))
        @test r2.reject isa AbstractDict
        @test r2.n_failed == r.n_failed
    end

    @testset "opp_sequence continuation from reloaded OPPResult" begin
        r2 = _roundtrip(extra.opp_pt)
        rng = MersenneTwister(7834)
        H = extra.ce.H
        fcs = [extra.fc,
               PolicyForecast{Float64}(extra.fc.outcomes,
                                       [randn(rng, H), randn(rng, H)],
                                       nothing, H, "next")]
        sq = opp_sequence(fcs, extra.ce, extra.loss)
        @test sq.delta[:, 1] ≈ r2.delta atol = 1e-10
        sq2 = _assert_roundtrip(sq)
        _assert_consumers(sq, sq2)
        @test sq2.delta[:, 1] ≈ r2.delta atol = 1e-10
    end

    @testset "model_average over reloaded ModelBankMember" begin
        mb = extra.mb
        menus2 = [PolicyCausalEffects(outcomes=[:pi], Theta_x=[1.1 .* ones(4, 4)],
                                      source=:dsge),
                  PolicyCausalEffects(outcomes=[:pi], Theta_x=[0.8 .* ones(4, 4)],
                                      source=:dsge)]
        mb_alt = ModelBankMember{Float64}("behavioral", [:kappa],
                                          reshape([0.11, 0.13], 2, 1),
                                          [-1.2, -1.3], -11.0, menus2, 0.28, 3)
        payload = _MEM._to_serializable(mb)
        @test payload["menu_draws"][1]["__struct__"] == "PolicyCausalEffects"
        orig = _MEM._suppress_warnings() do
            model_average([mb, mb_alt], [0.6, 0.4]; n_pool=30,
                          rng=MersenneTwister(7835))
        end
        reloaded = _MEM._suppress_warnings() do
            model_average([_roundtrip(mb), _roundtrip(mb_alt)], [0.6, 0.4];
                          n_pool=30, rng=MersenneTwister(7835))
        end
        @test reloaded isa PolicyCausalEffects{Float64}
        @test reloaded.source === :pooled
        @test _deep_equal(orig, reloaded)
        _assert_consumers(orig, reloaded)
    end

    @testset "CounterfactualMoments freq_band Tuple" begin
        cm = extra.cm_band
        @test cm.freq_band isa Tuple
        cm2 = _assert_roundtrip(cm)
        @test cm2.freq_band isa Tuple
        @test cm2.freq_band == cm.freq_band
        _assert_consumers(cm, cm2)
    end

    @testset "disk JLD2 OPPResult / ModelBankMember" begin
        dir = mktempdir()
        for (key, fname) in (("OPPResult", "opp.jld2"),
                             ("ModelBankMember", "bank.jld2"))
            r = byname[key]
            _MEM._assert_plain_payload(_MEM._build_container(r))
            path = joinpath(dir, fname)
            save_model(r, path)
            r3 = load_model(path)
            @test typeof(r3).name === typeof(r).name
            _assert_report_equal(r, r3)
            applicable(plot_result, r) && _assert_plot_equal(r, r3)
        end
    end
end
