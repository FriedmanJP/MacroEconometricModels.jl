# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-02 ThreeSLSModel serialization" begin
    pd = load_example(:grunfeld)
    ge = group_data(pd, "General Electric")
    wh = group_data(pd, "Westinghouse")
    n = 20
    y1 = ge.data[:, 1]; X1 = hcat(ones(n), ge.data[:, 2], ge.data[:, 3])
    y2 = wh.data[:, 1]; X2 = hcat(ones(n), wh.data[:, 2], wh.data[:, 3])
    Z = hcat(ones(n), ge.data[:, 3], wh.data[:, 3], wh.data[:, 2])
    m = estimate_3sls([(y1, X1, ["const", "value", "capital"]),
                       (y2, X2, ["const", "value", "capital"])], Z)

    args = Any[getfield(m, i) for i in 1:nfields(m)]
    @test _MEM._infer_float_param(args) === Float64
    @test _MEM._infer_float_param((m.betas,)) === Float64

    m2 = _assert_roundtrip(m)
    @test m2 isa ThreeSLSModel{Float64}
    @test m2.betas isa Vector{Vector{Float64}}
    @test eltype(eltype(m2.betas)) === Float64
    _assert_report_equal(m, m2)
    @test plot_result(m2) isa PlotOutput
    @test sprint(io -> refs(io, m)) == sprint(io -> refs(io, m2))
end
