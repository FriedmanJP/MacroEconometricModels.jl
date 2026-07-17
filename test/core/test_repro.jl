# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using Random
using LinearAlgebra

const _MEM = MacroEconometricModels

@testset "Reproducibility manifest (T246/#345)" begin

    @testset "capture_manifest populates the environment and never throws" begin
        m = capture_manifest()
        @test m isa ReproManifest
        @test m.seed === nothing
        @test m.n_threads == Threads.nthreads()
        @test m.julia_version == string(VERSION)
        @test !isempty(m.package_version)
        @test haskey(m.dependency_versions, "Distributions")
        @test haskey(m.dependency_versions, "StatsAPI")
        @test !isempty(m.timestamp)
        @test m.git_sha isa String              # a sha or "unknown", never an error
        @test m.git_dirty isa Bool
        @test m.os == string(Sys.KERNEL)

        m2 = capture_manifest(; seed=42, settings=Dict{String,Any}("reps" => 100))
        @test m2.seed == 42
        @test m2.settings["reps"] == 100
    end

    @testset "manifest ↔ dict round-trip (serialization bridge)" begin
        m = capture_manifest(; seed=7, settings=Dict{String,Any}("burnin" => 200, "thin" => 2))
        d = _MEM._manifest_to_dict(m)
        @test d["__manifest__"] == true
        m2 = _MEM._manifest_from_dict(d)
        @test m2.seed == m.seed
        @test m2.n_threads == m.n_threads
        @test m2.julia_version == m.julia_version
        @test m2.package_version == m.package_version
        @test m2.git_sha == m.git_sha
        @test m2.settings["burnin"] == 200
        @test m2.dependency_versions == m.dependency_versions
        @test _MEM._manifest_to_dict(nothing) === nothing
        @test _MEM._manifest_from_dict(nothing) === nothing
    end

    @testset "BVAR posterior carries a manifest and reproduces bit-for-bit" begin
        Y = randn(MersenneTwister(1), 80, 2)
        post = estimate_bvar(Y, 2; n_draws=100, seed=20260717)
        @test post.manifest isa ReproManifest
        @test post.manifest.seed == 20260717

        rep = reproduce(post)
        @test rep isa ReproReport
        @test rep.matched === true
        @test occursin("matched", rep.note)

        # same seed ⇒ identical draws; different seed ⇒ different draws
        post_same = estimate_bvar(Y, 2; n_draws=100, seed=20260717)
        @test post.B_draws == post_same.B_draws
        @test post.Sigma_draws == post_same.Sigma_draws
        post_diff = estimate_bvar(Y, 2; n_draws=100, seed=999)
        @test post.B_draws != post_diff.B_draws
    end

    @testset "BVAR gibbs sampler reproduces (burnin/thin recorded)" begin
        Y = randn(MersenneTwister(5), 70, 2)
        post = estimate_bvar(Y, 2; n_draws=60, sampler=:gibbs, thin=2, seed=314)
        @test post.manifest.settings["thin"] == 2
        @test post.manifest.settings["burnin"] == 200   # gibbs default recorded
        @test reproduce(post).matched === true
    end

    @testset "BVAR without a seed: manifest present, reproduction declines" begin
        Y = randn(MersenneTwister(2), 80, 2)
        post = estimate_bvar(Y, 2; n_draws=50)          # no seed
        @test post.manifest isa ReproManifest
        @test post.manifest.seed === nothing
        rep = reproduce(post)
        @test rep.matched === missing
        @test occursin("no recorded seed", rep.note)
    end

    @testset "bootstrap IRF carries a manifest and reproduces via reproduce(ir, model)" begin
        Y = randn(MersenneTwister(3), 100, 2)
        model = estimate_var(Y, 2)
        ir = irf(model, 10; ci_type=:bootstrap, reps=80, seed=123)
        @test ir.manifest isa ReproManifest
        @test ir.manifest.seed == 123
        @test ir.manifest.settings["reps"] == 80
        @test ir.manifest.settings["method"] == "cholesky"

        rep = reproduce(ir, model)
        @test rep.matched === true

        # single-arg form asks for the source model rather than throwing
        rep1 = reproduce(ir)
        @test rep1.matched === missing
        @test occursin("source model", rep1.note)

        # a deterministic (no-CI) IRF has no manifest
        ir_none = irf(model, 10)
        @test ir_none.manifest === nothing
    end

    @testset "thread-count caveat wording in the report note" begin
        # A fabricated manifest whose thread count differs from the current one
        # exercises both caveat branches deterministically.
        m = _MEM.ReproManifest(1, Threads.nthreads() + 1, "v", "p",
                               Dict{String,String}(), "os", "mach", "ts", "sha", false,
                               Dict{String,Any}())
        matched = _MEM._finalize_repro([_MEM.ReproFieldDiff("x", true, 0.0)], m)
        @test matched.matched === true
        @test occursin("thread-count-invariant", matched.note)

        mismatched = _MEM._finalize_repro([_MEM.ReproFieldDiff("x", false, 1.0)], m)
        @test mismatched.matched === false
        @test occursin("thread count changed", mismatched.note)
    end

    @testset "generic reproduce fallback for unsupported types" begin
        rep = reproduce(42)
        @test rep.matched === missing
        @test occursin("not implemented", rep.note)
    end

    @testset "reproducibility footer appears in show output" begin
        Y = randn(MersenneTwister(4), 80, 2)
        post = estimate_bvar(Y, 2; n_draws=50, seed=5)
        s = sprint(show, post)
        @test occursin("Reproducibility:", s)
        @test occursin("seed=5", s)

        post_ns = estimate_bvar(Y, 2; n_draws=50)
        s_ns = sprint(show, post_ns)
        @test occursin("seed=unset", s_ns)

        # ReproManifest and ReproReport render without error
        @test occursin("ReproManifest", sprint(show, post.manifest))
        @test occursin("ReproReport", sprint(show, reproduce(post)))
    end
end
