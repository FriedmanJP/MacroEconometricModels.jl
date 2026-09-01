# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const _RSER03 = ("ImpulseResponse", "BayesianImpulseResponse", "FEVD", "BayesianFEVD",
                 "HistoricalDecomposition", "BayesianHistoricalDecomposition",
                 "LPImpulseResponse", "LPFEVD", "StructuralLP",
                 "GrangerCausalityResult", "VARStationarityResult",
                 "PVARStability", "PVARTestResult",
                 "VECMGrangerResult", "VECMRestrictionTest")

const _RSER03_REPS = FAST ? 2 : 20
const _RSER03_H = FAST ? 4 : 8
const _RSER03_DRAWS = FAST ? 4 : 24

function _rser03_panel(; N=FAST ? 12 : 30, T_total=FAST ? 8 : 15, m=2,
                       rng=MersenneTwister(7763))
    data_mat = randn(rng, N * T_total, m)
    df = DataFrame(data_mat, ["y$i" for i in 1:m])
    df.id = repeat(1:N, inner=T_total)
    df.time = repeat(1:T_total, outer=N)
    xtset(df, :id, :time)
end

@testset "RSER-03 innovation-accounting serialization (#776)" begin
    @testset "registry" begin
        for name in _RSER03
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
        @test !any(v == "pending RSER-03" for v in values(_MEM._SERIALIZATION_EXCLUDED))
    end

    Y = randn(MersenneTwister(776), FAST ? 40 : 80, 2)
    m = estimate_var(Y, 2)

    @testset "ImpulseResponse bootstrap" begin
        ir = irf(m, _RSER03_H; ci_type=FAST ? :none : :bootstrap,
                 seed=1, reps=_RSER03_REPS)
        args = Any[getfield(ir, i) for i in 1:nfields(ir)]
        @test _MEM._infer_float_param(args) === Float64
        rebuilt = _MEM._generic_construct(ImpulseResponse, args)
        @test rebuilt isa ImpulseResponse{Float64}
        @test _from_serializable_is_generic(ImpulseResponse)
        ir2 = _assert_roundtrip(ir)
        _assert_consumers(ir, ir2)
        if !FAST
            @test ir.manifest isa ReproManifest
            @test ir.manifest.seed == 1
            @test rebuilt.manifest isa ReproManifest
            @test rebuilt.manifest.seed == 1
            @test reproduce(ir2, m).matched === true
            let path = joinpath(mktempdir(), "irf_boot.jld2")
                save_model(ir, path)
                ir3 = load_model(path)
                @test ir3 isa ImpulseResponse{Float64}
                _assert_report_equal(ir, ir3)
                _assert_plot_equal(ir, ir3)
                @test ir3.manifest isa ReproManifest
                @test ir3.manifest.seed == 1
                @test reproduce(ir3, m).matched === true
            end
        end
    end

    @testset "FEVD / HistoricalDecomposition / Granger / stationarity" begin
        fv = fevd(m, _RSER03_H)
        args_fv = Any[getfield(fv, i) for i in 1:nfields(fv)]
        @test _MEM._infer_float_param(args_fv) === Float64
        _assert_consumers(fv, _assert_roundtrip(fv))

        hd = historical_decomposition(m)
        args_hd = Any[getfield(hd, i) for i in 1:nfields(hd)]
        @test _MEM._infer_float_param(args_hd) === Float64
        hd2 = _assert_roundtrip(hd)
        _assert_consumers(hd, hd2)
        if !FAST
            let path = joinpath(mktempdir(), "hd.jld2")
                save_model(hd, path)
                hd3 = load_model(path)
                @test hd3 isa HistoricalDecomposition{Float64}
                _assert_report_equal(hd, hd3)
            end
        end

        g = granger_test(m, 1, 2)
        _assert_consumers(g, _assert_roundtrip(g))

        st = is_stationary(m)
        @test st isa VARStationarityResult
        st2 = _assert_roundtrip(st)
        _assert_consumers(st, st2)
        @test st2.is_stationary == st.is_stationary
        @test st2.max_modulus == st.max_modulus
    end

    FAST && return
    @testset "BayesianImpulseResponse / BayesianFEVD / BayesianHD" begin
        post = estimate_bvar(Y, 2; n_draws=_RSER03_DRAWS, seed=2)
        bir = irf(post, _RSER03_H)
        bir2 = _assert_roundtrip(bir)
        _assert_consumers(bir, bir2)
        if !FAST
            let path = joinpath(mktempdir(), "bir.jld2")
                save_model(bir, path)
                bir3 = load_model(path)
                @test bir3 isa BayesianImpulseResponse{Float64}
                _assert_report_equal(bir, bir3)
                _assert_plot_equal(bir, bir3)
            end
        end

        bfv = fevd(post, _RSER03_H)
        args_bfv = Any[getfield(bfv, i) for i in 1:nfields(bfv)]
        @test _MEM._infer_float_param(args_bfv) === Float64
        _assert_consumers(bfv, _assert_roundtrip(bfv))

        bhd = historical_decomposition(post)
        @test bhd isa BayesianHistoricalDecomposition
        _assert_consumers(bhd, _assert_roundtrip(bhd))
    end

    @testset "LPImpulseResponse / LPFEVD / StructuralLP" begin
        lp = estimate_lp(Y, 1, FAST ? 4 : 6; lags=1)
        lir = lp_irf(lp)
        _assert_consumers(lir, _assert_roundtrip(lir))

        slp = structural_lp(Y, FAST ? 4 : 6; method=:cholesky, lags=1, var_lags=2)
        slp2 = _assert_roundtrip(slp)
        _assert_consumers(slp, slp2)
        @test slp2.var_model isa VARModel{Float64}
        @test slp2.irf isa ImpulseResponse{Float64}
        @test slp2.lp_models isa Vector{<:LPModel}
        @test length(slp2.lp_models) == length(slp.lp_models)

        lf = lp_fevd(slp, FAST ? 4 : 6; method=:r2, n_boot=0)
        _assert_consumers(lf, _assert_roundtrip(lf))
    end

    @testset "VECM Granger + restriction test" begin
        Yci = cumsum(Y; dims=1)
        vecm = estimate_vecm(Yci, 2; rank=1)
        vg = granger_causality_vecm(vecm, 1, 2)
        @test vg isa VECMGrangerResult
        _assert_consumers(vg, _assert_roundtrip(vg))

        H = reshape([1.0, 0.0], 2, 1)
        vr = test_beta_restriction(vecm, H)
        @test vr isa VECMRestrictionTest
        vr2 = _assert_roundtrip(vr)
        _assert_consumers(vr, vr2)
        @test vr2.restricted_model isa VECMModel{Float64}
        @test vr2.kind === :beta
        @test irf(vr2.restricted_model, 4).values == irf(vr.restricted_model, 4).values
    end

    @testset "PVARStability / PVARTestResult" begin
        pd = _rser03_panel()
        pv = estimate_pvar(pd, 1; steps=:twostep, collapse=true)
        stab = pvar_stability(pv)
        _assert_consumers(stab, _assert_roundtrip(stab))

        j = pvar_hansen_j(pv)
        _assert_consumers(j, _assert_roundtrip(j))
        @test j.test_name == "Hansen J-test"
    end
end

@testset "RSER-04 VAR / conditional forecast serialization (#777)" begin
    Y = randn(MersenneTwister(777), FAST ? 40 : 80, 2)
    m = estimate_var(Y, 1)

    @testset "VARForecast" begin
        fc = forecast(m, FAST ? 4 : 6;
                      ci_method=FAST ? :analytic : :bootstrap,
                      reps=_RSER03_REPS, rng=MersenneTwister(1))
        if !FAST
            @test fc._draws isa Array{Float64,3}
            payload = _MEM._capture_fields(fc)
            @test haskey(payload, "_draws")
            @test payload["_draws"] !== nothing
        end
        @test _from_serializable_is_generic(VARForecast)
        fc2 = _assert_roundtrip(fc)
        _assert_consumers(fc, fc2)
        @test long_table(fc2) isa DataFrame
        @test fc2._draws == fc._draws
    end

    @testset "ForecastCondition / ConditionalForecast" begin
        cond = forecast_condition(1, 1, 0.5)
        @test _from_serializable_is_generic(ForecastCondition)
        cond2 = _assert_roundtrip(cond)
        _assert_report_equal(cond, cond2)
        @test cond2.variable == 1 && cond2.horizon == 1 && cond2.value == 0.5

        cf = conditional_forecast(m, [cond], 4; reps=_RSER03_REPS, rng=MersenneTwister(2))
        @test _from_serializable_is_generic(ConditionalForecast)
        cf2 = _assert_roundtrip(cf)
        _assert_consumers(cf, cf2)
        @test cf2.conditions isa Vector{<:ForecastCondition}
        @test length(cf2.conditions) == 1
        @test cf2.conditions[1].horizon == 1
        @test cf2.conditions[1].value == cf.conditions[1].value
        @test cf2.identification === cf.identification
    end
end
