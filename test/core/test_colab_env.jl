# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Contract tests for the Colab-only precompiled environment (#610).
# These check the in-repo recipe, not a built depot tarball (that lives on the
# GitHub Release and is produced by .github/workflows/colab-precompiled-env.yml).

using Test

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const COLAB_DIR = joinpath(REPO_ROOT, "environments", "colab")
const COLAB_PIN = "1.12.6"

@testset "Colab precompiled env (#610)" begin
    @testset "recipe files exist" begin
        @test isdir(COLAB_DIR)
        @test isfile(joinpath(COLAB_DIR, "Project.toml"))
        @test isfile(joinpath(COLAB_DIR, "Manifest.toml"))
        @test isfile(joinpath(COLAB_DIR, "colab_precompile.jl"))
        @test isfile(joinpath(COLAB_DIR, "README.md"))
        @test isfile(joinpath(COLAB_DIR, "colab_setup.ipynb"))
        @test isfile(joinpath(COLAB_DIR, "build_depot.sh"))
        @test isfile(joinpath(REPO_ROOT, ".github", "workflows", "colab-precompiled-env.yml"))
    end

    @testset "Project.toml pins the package UUID" begin
        proj = read(joinpath(COLAB_DIR, "Project.toml"), String)
        @test occursin("14a6ec33-bcac-448e-845f-2fb6769698f1", proj)
        @test occursin("MacroEconometricModels", proj)
        @test occursin("DataFrames", proj)
    end

    @testset "README is Colab-only and names the pin" begin
        readme = read(joinpath(COLAB_DIR, "README.md"), String)
        @test occursin("Google Colab only", readme)
        @test occursin("linux-x86_64", readme)
        @test occursin(COLAB_PIN, readme)
        @test occursin("JULIA_DEPOT_PATH", readme) || occursin("DEPOT_PATH", readme)
        @test occursin("PrecompileTools", readme)
    end

    @testset "workflow is tag/dispatch only" begin
        wf = read(joinpath(REPO_ROOT, ".github", "workflows", "colab-precompiled-env.yml"), String)
        @test occursin("workflow_dispatch", wf)
        @test occursin(COLAB_PIN, wf)
        @test occursin("tags:", wf)
        # Must not run on every PR (cost control).
        @test !occursin(r"(?m)^  pull_request:", wf)
    end

    @testset "colab_precompile.jl is parseable and seeded" begin
        src = read(joinpath(COLAB_DIR, "colab_precompile.jl"), String)
        @test occursin("MersenneTwister", src)
        @test occursin("estimate_var", src)
        @test occursin("estimate_reg", src)
        @test occursin("estimate_bvar", src)
        @test occursin("estimate_lp", src)
        Meta.parse("begin\n" * src * "\nend")
    end
end
