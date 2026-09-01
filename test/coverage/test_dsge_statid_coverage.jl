# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Codecov for DSGE identification / priors / MCMC diagnostics / prefilter
# (src/dsge/identification.jl, priors.jl, mcmc_diagnostics.jl, prefilter.jl,
# family_facades.jl helpers). FAST-friendly: no SMC, no long PF.

using Test
using MacroEconometricModels
using LinearAlgebra
using Random
using Statistics
using Distributions

const M = MacroEconometricModels

@testset "DSGE statid / prior / MCMC coverage" begin

    @testset "InverseGamma1 + dynare_prior" begin
        d = M.InverseGamma1(2.0, 5.0)
        @test minimum(d) == 0.0
        @test isinf(maximum(d))
        @test insupport(d, 0.5) && !insupport(d, -0.1)
        @test pdf(d, -1.0) == 0.0
        @test isfinite(pdf(d, 1.0))
        @test cdf(d, -1.0) == 0.0
        @test 0 < cdf(d, 1.0) < 1
        q = quantile(d, 0.5)
        @test q > 0
        @test isfinite(mean(d)) && isfinite(var(d)) && isfinite(std(d))
        @test rand(MersenneTwister(1), d) > 0
        d_nan = M.InverseGamma1(1.0, 0.5)
        @test isnan(mean(d_nan))
        @test isnan(var(M.InverseGamma1(1.0, 1.5)))
        @test_throws ArgumentError M.InverseGamma1(-1.0, 4.0)
        @test_throws ArgumentError M.InverseGamma1(1.0, -4.0)

        @test dynare_prior(:normal, 0.3, 0.05) isa Normal
        @test dynare_prior(:gamma, 1.5, 0.25) isa Gamma
        @test dynare_prior(:beta, 0.7, 0.1) isa Beta
        gb = dynare_prior(:beta, 0.5, 0.1; lower=0.0, upper=0.9)
        @test gb isa Distribution
        ig1 = dynare_prior(:inv_gamma, 0.02, 0.05)
        @test ig1 isa M.InverseGamma1
        @test dynare_prior(:inv_gamma1, 0.02, 0.05) isa M.InverseGamma1
        @test dynare_prior(:inv_gamma2, 0.04, 0.02) isa InverseGamma
        @test dynare_prior(:uniform, 0.0, 0.2) isa Uniform
        @test dynare_prior(:uniform, 0.0, 0.1; lower=-1.0, upper=1.0) isa Uniform
        @test_throws ArgumentError dynare_prior(:normal, 0.0, 0.0)
        @test_throws ArgumentError dynare_prior(:gamma, -1.0, 0.1)
        @test_throws ArgumentError dynare_prior(:beta, 0.5, 0.1; lower=1.0, upper=0.0)
        @test_throws ArgumentError dynare_prior(:beta, 2.0, 0.1)
        @test_throws ArgumentError dynare_prior(:uniform, 0.0, 0.1; lower=0.0, upper=nothing)
        @test_throws ArgumentError dynare_prior(:bogus, 0.0, 1.0)
    end

    @testset "identification_diagnostics (Iskrev)" begin
        spec = @dsge begin
            parameters: rho = 0.8, sigma = 0.1
            endogenous: y
            exogenous: e
            y[t] = rho * y[t-1] + sigma * e[t]
        end
        spec = compute_steady_state(spec)
        idd = identification_diagnostics(spec, [:rho, :sigma]; n_lags=2)
        @test idd isa M.IdentificationDiagnostics
        @test idd.n_params == 2
        @test idd.n_lags == 2
        @test length(idd.singular_values) >= 1
        s = sprint(show, idd)
        @test occursin("Iskrev", s)
        @test_throws ArgumentError identification_diagnostics(spec, Symbol[])
        @test_throws ArgumentError identification_diagnostics(spec, [:rho]; theta=[0.1, 0.2])
        @test_throws ArgumentError identification_diagnostics(spec, [:rho]; observables=[:nope])
    end

    @testset "MCMC diagnostic internals" begin
        rng = MersenneTwister(3)
        x = randn(rng, 80)
        x[1:5] .= x[1]                       # ties for _tied_ranks
        r = M._tied_ranks(x)
        @test length(r) == 80
        @test extrema(r)[1] >= 1
        z = M._rank_normalize(reshape(x, 40, 2))
        @test size(z) == (40, 2)
        sc = M._split_chain(x)
        @test size(sc, 2) == 2
        @test isfinite(M._rhat_rank(x))
        @test isfinite(M._ess_bulk(x))
        @test isfinite(M._ess_tail(x))
        @test isfinite(M._geweke_nse(x))
        @test isnan(M._rhat_rank(randn(4)))
        @test isnan(M._ess_bulk(randn(4)))
        @test isnan(M._rhat_chains(randn(3, 2)))
        @test isnan(M._ess_chains(randn(3, 2)))
    end

    @testset "detect_trend / PrefilterSpec show" begin
        @test detect_trend(randn(5)).trending == false
        tr = collect(0.0:0.5:20.0)
        d = detect_trend(tr)
        @test d.trending
        flags = detect_trend(hcat(tr, randn(length(tr))); names=[:trend, :noise], warn=true)
        @test flags[1] && flags[2] == false
        Y = randn(1, 40)   # n_obs × T (Kalman orientation)
        _, pf = apply_prefilter(Y, :demean; observables=[:y])
        @test occursin("PrefilterSpec", sprint(show, pf))
        @test occursin(":demean", sprint(show, pf))
    end

    @testset "family facade helpers" begin
        ir = M._path_to_irf(randn(6, 2), ("y", "c"), "e")
        @test ir isa ImpulseResponse
        @test size(ir.values, 1) == 6
        fv = M._fevd_from_irf(ir)
        @test fv isa FEVD
        M._warn_mit_terminal(0.0, 10)
        M._warn_mit_terminal(0.99, 3)   # should warn: tail not decayed
        sk, rest = M._split_solve_kwargs((method=:ssj, shock_size=0.02, horizon=10))
        @test haskey(sk, :method)
        @test haskey(rest, :shock_size)
        @test_throws ArgumentError M._ct_z_path(1.0, 1, 0.01, 0.0)
        z = M._ct_z_path(1.0, 4, 0.01, 0.0)
        @test length(z) == 4
    end

end
