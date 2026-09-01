# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-02 JohansenResult serialization" begin
    Yci = cumsum(randn(MersenneTwister(3), 100, 2); dims=1)
    jr = johansen_test(Yci, 2)
    jr2 = _assert_roundtrip(jr)
    _assert_report_equal(jr, jr2)
    @test plot_result(jr2) isa PlotOutput
    @test sprint(io -> refs(io, jr)) == sprint(io -> refs(io, jr2))
end
