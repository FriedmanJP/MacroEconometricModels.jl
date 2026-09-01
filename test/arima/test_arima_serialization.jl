# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-02 ARIMA / ARDL / long-memory serialization" begin
    ya = randn(MersenneTwister(11), 160)
    xa = cumsum(randn(MersenneTwister(41), 160))

    @testset "SARIMAModel" begin
        m = estimate_sarima(ya, 1, 0, 1, 0, 0, 0, 4; method=:css)
        m2 = _assert_roundtrip(m)
        _assert_report_equal(m, m2)
        @test plot_result(m2) isa PlotOutput
        @test coef(m2) == coef(m)
        @test stderror(m2) == stderror(m)
        @test vcov(m2) == vcov(m)
        @test fitted(m2) == fitted(m)
        f1, f2 = forecast(m, 6), forecast(m2, 6)
        @test f1.forecast == f2.forecast && f1.se == f2.se
        @test sprint(io -> refs(io, m)) == sprint(io -> refs(io, m2))
        let path = joinpath(mktempdir(), "sarima.jld2")
            save_model(m, path)
            m3 = load_model(path)
            @test m3 isa SARIMAModel{Float64}
            @test sprint(show, m3) == sprint(show, m)
            @test forecast(m3, 6).forecast == f1.forecast
        end
    end

    @testset "ARIMAOrderSelection nested models" begin
        sel = select_arima_order(ya, 1, 1; d=0, method=:css)
        sel2 = _assert_roundtrip(sel)
        _assert_report_equal(sel, sel2)
        @test plot_result(sel2) isa PlotOutput
        @test sel2.best_model_aic isa typeof(sel.best_model_aic)
        @test sel2.best_model_bic isa typeof(sel.best_model_bic)
        @test sel2.best_model_aic isa AbstractARIMAModel
        @test !(typeof(sel2.best_model_aic) <: AbstractARIMAModel &&
                nameof(typeof(sel2.best_model_aic)) === :AbstractARIMAModel)

        sar = estimate_sarima(ya, 1, 0, 1, 0, 0, 0, 4; method=:css)
        nested = ARIMAOrderSelection(1, 1, 1, 1,
                                     fill(sar.aic, 2, 2), fill(sar.bic, 2, 2),
                                     sar, sar)
        nested2 = _assert_roundtrip(nested)
        @test nested2.best_model_aic isa SARIMAModel{Float64}
        @test nested2.best_model_bic isa SARIMAModel{Float64}
        _assert_report_equal(nested, nested2)
    end

    @testset "GPHResult / LocalWhittleResult" begin
        g = gph_test(ya)
        g2 = _assert_roundtrip(g)
        _assert_report_equal(g, g2)
        @test sprint(io -> refs(io, g)) == sprint(io -> refs(io, g2))

        lw = local_whittle(ya)
        lw2 = _assert_roundtrip(lw)
        _assert_report_equal(lw, lw2)
        @test sprint(io -> refs(io, lw)) == sprint(io -> refs(io, lw2))
    end

    @testset "ARDLLongRun / ARDLBoundsTest" begin
        ardl = estimate_ardl(ya, reshape(xa, :, 1); p=1, q=1, case=3)
        lr = long_run(ardl)
        lr2 = _assert_roundtrip(lr)
        _assert_report_equal(lr, lr2)
        @test lr2.theta == lr.theta && lr2.se == lr.se

        bt = bounds_test(ardl)
        bt2 = _assert_roundtrip(bt)
        _assert_report_equal(bt, bt2)
        @test bt2.f_decision === bt.f_decision
        ardl2 = _roundtrip(ardl)
        @test _deep_equal(long_run(ardl2), long_run(ardl))
        @test _deep_equal(bounds_test(ardl2), bounds_test(ardl))
    end

    @testset "NARDLMultipliers / NARDLSymmetryTest" begin
        nm = estimate_nardl(ya, reshape(xa, :, 1); p=1, q=1)
        st = symmetry_test(nm)
        st2 = _assert_roundtrip(st)
        _assert_report_equal(st, st2)
        @test sprint(io -> refs(io, st)) == sprint(io -> refs(io, st2))

        mm = dynamic_multipliers(nm, 6; bootstrap=false)
        mm2 = _assert_roundtrip(mm)
        _assert_report_equal(mm, mm2)
        @test plot_result(mm2) isa PlotOutput

        nm2 = _roundtrip(nm)
        @test _deep_equal(symmetry_test(nm2), st)
        @test _deep_equal(dynamic_multipliers(nm2, 6; bootstrap=false), mm)
        @test _deep_equal(bounds_test(nm2), bounds_test(nm))
    end
end
