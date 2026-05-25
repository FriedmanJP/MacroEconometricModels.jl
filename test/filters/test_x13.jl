using Test
using Random
using Statistics
using LinearAlgebra

@testset "X-13ARIMA-SEATS Seasonal Adjustment" begin

    Random.seed!(42)
    n = 120
    trend_c = cumsum(randn(n) .* 0.1)
    seasonal_c = 10.0 .* sin.(2π .* (1:n) ./ 12) .+ 5.0 .* cos.(2π .* (1:n) ./ 6)
    y = 100.0 .+ trend_c .+ seasonal_c .+ randn(n)

    @testset "basic X-11 functionality" begin
        r = x13_filter(y; frequency=12, method=:x11)
        @test r isa X13FilterResult{Float64}
        @test r.method == :x11
        @test r.T_obs == n
        @test r.frequency == 12
        @test length(r.trend) == n
        @test length(r.seasonal) == n
        @test length(r.irregular) == n
        @test length(r.adjusted) == n
        @test length(r.original) == n
        @test r.original ≈ y
        @test r.arima_order isa NTuple{6,Int}
    end

    @testset "basic SEATS functionality" begin
        r = x13_filter(y; frequency=12, method=:seats)
        @test r isa X13FilterResult{Float64}
        @test r.method == :seats
        @test r.T_obs == n
        @test length(r.trend) == n
        @test length(r.seasonal) == n
        @test length(r.adjusted) == n
    end

    @testset "seasonally adjusted preserves trend" begin
        r = x13_filter(y; frequency=12, method=:x11)
        sa_std = std(diff(r.adjusted))
        orig_std = std(diff(y))
        @test sa_std < orig_std
    end

    @testset "seasonal component captures periodicity" begin
        r = x13_filter(y; frequency=12, method=:x11)
        @test std(r.seasonal) > 0.01
        @test std(r.irregular) < std(y)
    end

    @testset "unified accessors" begin
        r = x13_filter(y; frequency=12, method=:x11)
        @test trend(r) === r.trend
        @test cycle(r) === r.irregular
        @test seasonal(r) === r.seasonal
        @test adjusted(r) === r.adjusted
    end

    @testset "quarterly data" begin
        Random.seed!(123)
        nq = 80
        yq = 100.0 .+ cumsum(randn(nq) .* 0.1) .+ 5.0 .* sin.(2π .* (1:nq) ./ 4) .+ randn(nq)
        r = x13_filter(yq; frequency=4, method=:x11)
        @test r isa X13FilterResult{Float64}
        @test r.frequency == 4
        @test r.T_obs == nq
        @test length(r.trend) == nq
    end

    @testset "log transformation" begin
        y_pos = exp.(3.0 .+ 0.01 .* (1:n) .+ 0.3 .* sin.(2π .* (1:n) ./ 12) .+ 0.1 .* randn(n))
        r = x13_filter(y_pos; frequency=12, transform=:log)
        @test r.transform == :log
        @test all(isfinite, r.trend)
        @test all(isfinite, r.seasonal)
    end

    @testset "no-transform option" begin
        r = x13_filter(y; frequency=12, transform=:none)
        @test r.transform == :none
    end

    @testset "ARIMA model identification" begin
        r = x13_filter(y; frequency=12)
        p, d, q, P, D, Q = r.arima_order
        @test p >= 0 && d >= 0 && q >= 0
        @test P >= 0 && D >= 0 && Q >= 0
        @test isfinite(r.sigma2)
        @test isfinite(r.aic)
    end

    @testset "Float32 input" begin
        y32 = Float32.(y)
        r = x13_filter(y32; frequency=12, method=:x11)
        @test r isa X13FilterResult{Float32}
        @test length(r.trend) == n
    end

    @testset "Integer input fallback" begin
        yi = round.(Int, y)
        r = x13_filter(yi; frequency=12, method=:x11)
        @test r isa X13FilterResult{Float64}
    end

    @testset "edge cases" begin
        @test_throws ArgumentError x13_filter(randn(20); frequency=12)
        @test_throws ArgumentError x13_filter(y; frequency=7)
        @test_throws ArgumentError x13_filter(y; frequency=12, method=:invalid)
    end

    @testset "display and refs" begin
        r = x13_filter(y; frequency=12, method=:x11)
        io = IOBuffer()
        show(io, r)
        s = String(take!(io))
        @test occursin("X-13ARIMA-SEATS", s)
        @test occursin("Observations", s)

        io2 = IOBuffer()
        refs(io2, r)
        s2 = String(take!(io2))
        @test occursin("Dagum", s2) || occursin("Findley", s2)
    end

    @testset "report" begin
        r = x13_filter(y; frequency=12, method=:x11)
        report(r)
    end

    @testset "plot_result" begin
        r = x13_filter(y; frequency=12, method=:x11)
        p = plot_result(r)
        @test p isa PlotOutput
    end

end
