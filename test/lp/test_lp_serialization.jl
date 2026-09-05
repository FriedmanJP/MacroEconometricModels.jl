# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-03 LP innovation-accounting serialization (#776)" begin
    Y = dgp_var(MersenneTwister(776); A=[0.5 0.1; 0.2 0.4], B0=[1.0 0.0; 0.0 1.0], T=80).Y

    @testset "LPImpulseResponse" begin
        lp = estimate_lp(Y, 1, 6; lags=1)
        lir = lp_irf(lp)
        lir2 = _assert_roundtrip(lir)
        _assert_consumers(lir, lir2)
        @test lir2.shock_var == lir.shock_var
        @test lir2.cov_type === lir.cov_type
    end

    @testset "StructuralLP nested VARModel / LPModel / ImpulseResponse" begin
        slp = structural_lp(Y, 6; method=:cholesky, lags=1, var_lags=2)
        slp2 = _assert_roundtrip(slp)
        _assert_consumers(slp, slp2)
        @test slp2.irf isa ImpulseResponse{Float64}
        @test slp2.var_model isa VARModel{Float64}
        @test slp2.lp_models isa Vector{<:LPModel}
        @test length(slp2.lp_models) == 2
        @test slp2.method === :cholesky
        @test slp2.Q == slp.Q
        @test irf(slp2.var_model, 4).values == irf(slp.var_model, 4).values
        @test coef(slp2.lp_models[1]) == coef(slp.lp_models[1])
    end

    @testset "LPFEVD" begin
        slp = structural_lp(Y, 6; method=:cholesky, lags=1, var_lags=2)
        lf = lp_fevd(slp, 6; method=:r2, n_boot=0)
        lf2 = _assert_roundtrip(lf)
        _assert_consumers(lf, lf2)
        @test lf2.method === :r2
        @test lf2.n_boot == 0
        @test lf2.bias_correction == lf.bias_correction
    end
end

@testset "RSER-04 LPForecast serialization (#777)" begin
    Y = dgp_var(MersenneTwister(777); A=[0.5 0.1; 0.2 0.4], B0=[1.0 0.0; 0.0 1.0], T=80).Y
    lp = estimate_lp(Y, 1, 6; lags=1)
    fc = forecast(lp, ones(6); ci_method=:analytical)
    @test _from_serializable_is_generic(LPForecast)
    fc2 = _assert_roundtrip(fc)
    _assert_consumers(fc, fc2)
    @test fc2.ci_method === :analytical
    @test fc2.shock_path == fc.shock_path
    @test long_table(fc2) isa DataFrame
end

const _RSER11_LP = ("BSplineBasis", "LPIVARBand", "MontielOleaPfluegerF",
                    "PropensityScoreConfig")

"""LP-IV DGP used by test_lp_weak_iv.jl: `pi1` sets instrument strength."""
function _rser11_lpiv_sim(T_obs::Int; pi1::Float64=1.5, seed::Int=784, theta::Float64=1.0)
    rng = MersenneTwister(seed)
    z = randn(rng, T_obs)
    v = randn(rng, T_obs)
    s = pi1 .* z .+ v
    y = zeros(T_obs)
    x2 = zeros(T_obs)
    for t in 2:T_obs
        y[t] = 0.5 * y[t-1] + theta * s[t] + 0.6 * v[t] + randn(rng)
        x2[t] = 0.3 * x2[t-1] + 0.4 * s[t] + randn(rng)
    end
    return hcat(s, y, x2), reshape(z, :, 1)
end

@testset "RSER-11 LP leftovers serialization (#784)" begin
    @testset "registry" begin
        for name in _RSER11_LP
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
    end

    @testset "MontielOleaPfluegerF" begin
        @test _from_serializable_is_generic(MontielOleaPfluegerF)
        Y, Z = _rser11_lpiv_sim(200)
        m = estimate_lp_iv(Y, 1, Z, 4; lags=2, varnames=["s", "y", "x2"])
        mop = montiel_olea_pflueger_f(m)
        @test mop isa MontielOleaPfluegerF{Float64}
        mop2 = _assert_roundtrip(mop)
        _assert_consumers(mop, mop2)
        @test mop2.tau == mop.tau
        @test mop2.f_effective == mop.f_effective
        @test mop2.weak == mop.weak
    end

    @testset "LPIVARBand nested Tuple eltype" begin
        @test _from_serializable_is_generic(LPIVARBand)
        Y, Z = _rser11_lpiv_sim(160)
        m = estimate_lp_iv(Y, 1, Z, 3; lags=1, varnames=["s", "y", "x2"])
        band = lp_iv_ar_band(m; responses=[2], n_grid=31)
        @test band isa LPIVARBand{Float64}
        @test band.sets isa Matrix{Vector{Tuple{Float64,Float64}}}
        band2 = _assert_roundtrip(band)
        _assert_consumers(band, band2)
        @test band2.sets isa Matrix{Vector{Tuple{Float64,Float64}}}
        @test eltype(band2.sets) === Vector{Tuple{Float64,Float64}}
        @test eltype(eltype(band2.sets)) === Tuple{Float64,Float64}
        @test band2.sets == band.sets
        @test band2.lower == band.lower
        payload = _MEM._to_serializable(band)
        @test payload["sets"] isa AbstractArray
        @test eltype(eltype(band2.sets[1, 1])) === Float64
    end

    @testset "BSplineBasis from SmoothLPModel" begin
        @test _from_serializable_is_generic(BSplineBasis)
        Y = dgp_var(MersenneTwister(7841); A=[0.5 0.1; 0.2 0.4], B0=[1.0 0.0; 0.0 1.0], T=80).Y
        sm = estimate_smooth_lp(Y, 1, 8; degree=3, n_knots=4, lambda=1.0, lags=1)
        basis = sm.spline_basis
        @test basis isa BSplineBasis{Float64}
        basis2 = _assert_roundtrip(basis)
        _assert_consumers(basis, basis2)
        @test basis2.degree == 3
        @test basis2.basis_matrix == basis.basis_matrix
        sm2 = _assert_roundtrip(sm)
        @test sm2.spline_basis isa BSplineBasis{Float64}
        @test sm2.spline_basis.knots == basis.knots
    end

    @testset "PropensityScoreConfig" begin
        @test _from_serializable_is_generic(PropensityScoreConfig)
        cfg = PropensityScoreConfig(method=:probit, trimming=(0.05, 0.95), normalize=false)
        @test cfg isa PropensityScoreConfig{Float64}
        cfg2 = _assert_roundtrip(cfg)
        _assert_consumers(cfg, cfg2)
        @test cfg2.method === :probit
        @test cfg2.trimming isa Tuple{Float64,Float64}
        @test cfg2.trimming === (0.05, 0.95)

        Tobs, n = 60, 2
        rng = MersenneTwister(7842)
        X = randn(rng, Tobs, 2)
        treatment = rand(rng, Tobs) .< 0.4
        Y = randn(rng, Tobs, n)
        plp = estimate_propensity_lp(Y, treatment, X, 3; ps_method=:logit, lags=1)
        live = _assert_roundtrip(plp.config)
        @test live.method === :logit
        @test live.trimming isa Tuple{Float64,Float64}
    end

    @testset "disk round-trip LPIVARBand" begin
        Y, Z = _rser11_lpiv_sim(120; seed=7843)
        m = estimate_lp_iv(Y, 1, Z, 2; lags=1)
        band = lp_iv_ar_band(m; responses=[2], n_grid=21)
        path = joinpath(mktempdir(), "lpivar.jld2")
        save_model(band, path)
        band2 = load_model(path)
        @test band2 isa LPIVARBand{Float64}
        @test band2.sets isa Matrix{Vector{Tuple{Float64,Float64}}}
        @test band2.sets == band.sets
    end
end

