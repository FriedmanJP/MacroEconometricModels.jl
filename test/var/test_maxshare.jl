# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

if !@isdefined(simulate_svar)
    include("id_dgps.jl")
end

const MEM = MacroEconometricModels

"""Population VAR with known companion coefficients (no sampling error)."""
function _known_var(A1::AbstractMatrix{T}, Sigma::AbstractMatrix{T};
                    Tobs::Int=40, varnames::Union{Nothing,Vector{String}}=nothing) where {T<:AbstractFloat}
    n = size(A1, 1)
    p = 1
    Y = zeros(T, Tobs, n)
    B = zeros(T, 1 + n * p, n)
    B[2:(1 + n), :] = A1'
    U = zeros(T, Tobs - p, n)
    vn = something(varnames, ["y$i" for i in 1:n])
    VARModel(Y, p, B, U, Matrix{T}(Sigma), zero(T), zero(T), zero(T), vn)
end

@testset "SID-12 max-share identification" begin
    A1 = Diagonal([0.85, 0.30, 0.15])
    Sigma = Matrix{Float64}(I, 3, 3)
    model = _known_var(Matrix{Float64}(A1), Sigma; varnames=["prod", "gdp", "hours"])

    @testset "registry flags and compute_Q" begin
        @test haskey(MEM.IDENTIFICATION_REGISTRY, :max_share)
        @test !MEM._needs_residuals(:max_share)
        @test !MEM._is_set_identified(:max_share)
        @test MEM._is_partial(:max_share)
        @test !MEM._should_match_columns(:max_share)
        Q = MEM.compute_Q(model, :max_share; target=1, horizons=0:20)
        @test size(Q) == (3, 3)
        @test norm(Q' * Q - I(3)) < 1e-8
        @test_throws ArgumentError MEM.compute_Q(model, :max_share)
    end

    @testset "identify_max_share returns MaxShareResult" begin
        r = identify_max_share(model; target=1, horizons=0:20)
        @test r isa MaxShareResult
        @test r.target == 1
        @test r.horizons == 0:20
        @test r.band === nothing
        @test r.is_partial
        @test size(r.Q) == (3, 3)
        @test length(r.q) == 3
        @test r.Q[:, 1] ≈ r.q
        @test norm(r.Q' * r.Q - I(3)) < 1e-10
        @test 0 < r.share <= 1
        @test length(r.eigvals) == 3
        @test r.share ≈ r.eigvals[1] / sum(r.eigvals) atol = 1e-10
        @test length(r.varnames) == 3
        @test length(r.shock_names) == 3
        @test occursin("Unidentified", r.shock_names[2])
    end

    @testset "string target and sign normalisation" begin
        r = identify_max_share(model; target="prod", horizons=0:8)
        @test r.target == 1
        L = cholesky_factor(model)
        @test (L * r.q)[1] > 0
    end

    @testset "FEVD share equals leading eigenvalue / total" begin
        Hwin = 0:12
        r = identify_max_share(model; target=1, horizons=Hwin)
        H = last(Hwin) + 1
        fv = fevd(model, H; method=:max_share, target=1, horizons=Hwin)
        @test fv.proportions[1, 1, H] ≈ r.share atol = 1e-8
        @test r.share ≈ r.eigvals[1] / sum(r.eigvals) atol = 1e-10
    end

    @testset "frequency band recovers the same shock" begin
        r_time = identify_max_share(model; target=1, horizons=0:200)
        r_freq = identify_max_share(model; target=1, band=(0.0, Float64(π)))
        @test r_freq.horizons === nothing
        @test r_freq.band !== nothing
        @test abs(dot(r_time.q, r_freq.q)) > 0.99
    end

    @testset "time and frequency mutually exclusive" begin
        @test_throws ArgumentError identify_max_share(model; target=1,
                                                      horizons=0:8, band=(0.1, 0.5))
    end

    @testset "sequential shocks are orthogonal to previous" begin
        Aoff = [0.5 0.35 0.0; 0.25 0.5 0.1; 0.0 0.15 0.4]
        moff = _known_var(Aoff, Sigma; varnames=["prod", "gdp", "hours"])
        surprise = identify_max_share(moff; target=1, horizons=0:0)
        news = identify_max_share(moff; target=1, horizons=0:20, previous=surprise.q)
        @test abs(dot(news.q, surprise.q)) < 1e-8
        one = identify_max_share(moff; target=1, horizons=0:20)
        two = identify_max_share(moff; target=1, horizons=0:20, n_shocks=2)
        @test abs(dot(two.Q[:, 1], one.q)) > 0.99
        @test abs(dot(two.Q[:, 1], two.Q[:, 2])) < 1e-8
    end

    @testset "irf/fevd/hd method=:max_share" begin
        ir = irf(model, 8; method=:max_share, target=1, horizons=0:8)
        @test ir isa ImpulseResponse
        @test size(ir.values) == (8, 3, 3)
        @test isfinite(ir.values[1, 1, 1])
        fv = fevd(model, 8; method=:max_share, target=1, horizons=0:8)
        @test fv isa FEVD
        hd = historical_decomposition(model, 20; method=:max_share, target=1, horizons=0:8)
        @test hd isa HistoricalDecomposition
    end

    @testset "report and refs" begin
        r = identify_max_share(model; target=1, horizons=0:8)
        report(r)
        refs(r)
        @test point_estimate(r) == r.Q
    end

    @testset "cumulative option is valid" begin
        r = identify_max_share(model; target=1, horizons=0:8, cumulative=false)
        rc = identify_max_share(model; target=1, horizons=0:8, cumulative=true)
        @test rc isa MaxShareResult
        @test abs(dot(r.q, rc.q)) > 0.99  # diagonal DGP: both recover e₁
        Aoff = [0.7 0.4 0.0; 0.2 0.5 0.1; 0.0 0.1 0.4]
        moff = _known_var(Aoff, Sigma)
        ro = identify_max_share(moff; target=1, horizons=0:8, cumulative=false)
        rc2 = identify_max_share(moff; target=1, horizons=0:8, cumulative=true)
        @test ro isa MaxShareResult && rc2 isa MaxShareResult
    end

    @testset "plot_result" begin
        r = identify_max_share(model; target=1, horizons=0:8)
        p = plot_result(r)
        @test p !== nothing
    end
end
