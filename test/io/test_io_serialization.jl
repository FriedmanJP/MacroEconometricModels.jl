# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

const _RSER11_IO = ("FootprintResult", "IOExtension", "IOMultipliers", "LinkageResult")

@testset "RSER-11 IO leftovers serialization (#784)" begin
    @testset "registry" begin
        for name in _RSER11_IO
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
        @test _MEM._SERIALIZATION_EXCLUDED["LeontiefModel"] == "internal IO coefficient wrapper"
        @test _MEM._SERIALIZATION_EXCLUDED["GhoshModel"] == "internal IO coefficient wrapper"
        @test _MEM._SERIALIZATION_EXCLUDED["IOSourceTable"] == "download registry, not a result"
        # Underscore-prefixed X-13 internals are skipped by completeness (RSER-08);
        # they are neither save targets nor exported, so they stay out of both dicts.
        @test !any(startswith(k, "_X13") for k in keys(_MEM._SERIALIZABLE_TYPES))
        @test !any(startswith(k, "_X13") for k in keys(_MEM._SERIALIZATION_EXCLUDED))
        @test :IOMultipliers in names(MacroEconometricModels)
        @test :LinkageResult in names(MacroEconometricModels)
    end

    io = load_example(:wiot)

    @testset "FootprintResult" begin
        @test _from_serializable_is_generic(FootprintResult)
        fp = footprint(io, "CO2")
        @test fp isa FootprintResult
        fp2 = _assert_roundtrip(fp)
        _assert_consumers(fp, fp2)
        @test fp2.name == "CO2"
        @test sum(fp2.total) ≈ sum(fp.total)
    end

    @testset "IOExtension standalone and nested in IOData" begin
        @test _from_serializable_is_generic(IOExtension)
        ext = io.extensions["CO2"]
        @test ext isa IOExtension
        ext2 = _assert_roundtrip(ext)
        @test ext2.stressors == ext.stressors
        @test ext2.unit == ext.unit
        io2 = _assert_roundtrip(io)
        @test io2.extensions["CO2"] isa IOExtension
        @test io2.extensions["CO2"].S == ext.S
    end

    @testset "IOMultipliers" begin
        @test _from_serializable_is_generic(IOMultipliers)
        om = multipliers(io; kind=:output, type=:I)
        @test om isa IOMultipliers
        om2 = _assert_roundtrip(om)
        _assert_consumers(om, om2)
        @test om2.kind === :output
        @test om2.type === :I
        @test om2.values == om.values
    end

    @testset "LinkageResult" begin
        @test _from_serializable_is_generic(LinkageResult)
        lk = linkages(io)
        @test lk isa LinkageResult
        lk2 = _assert_roundtrip(lk)
        _assert_consumers(lk, lk2)
        @test lk2.classification == lk.classification
        @test lk2.Ui == lk.Ui
    end

    @testset "disk round-trip FootprintResult / IOMultipliers" begin
        fp = footprint(io, "CO2")
        om = multipliers(io; kind=:income, type=:I)
        dir = mktempdir()
        save_model(fp, joinpath(dir, "fp.jld2"))
        save_model(om, joinpath(dir, "om.jld2"))
        @test load_model(joinpath(dir, "fp.jld2")) isa FootprintResult
        @test load_model(joinpath(dir, "om.jld2")).values == om.values
    end
end
