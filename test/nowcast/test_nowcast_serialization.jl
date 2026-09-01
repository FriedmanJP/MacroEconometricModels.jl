# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

const _RSER05 = ("NowcastDFM", "NowcastBVAR", "NowcastBridge",
                 "NowcastResult", "NowcastNews", "NowcastForecast")

# Mixed-frequency panel matching `_make_nowcast_data` in test_nowcast.jl.
function _rser05_data(; T_obs=60, nM=4, nQ=1, r=1, seed=778)
    rng = Random.MersenneTwister(seed)
    F = randn(rng, T_obs, r)
    for t in 2:T_obs
        F[t, :] = 0.7 * F[t-1, :] + 0.3 * randn(rng, r)
    end
    X_M = F * randn(rng, nM, r)' + 0.2 * randn(rng, T_obs, nM)
    X_Q = F * randn(rng, nQ, r)' + 0.2 * randn(rng, T_obs, nQ)
    for t in 1:T_obs
        if mod(t, 3) != 0
            X_Q[t, :] .= NaN
        end
    end
    return hcat(X_M, X_Q)
end

function _assert_plot_view_equal(a, b, view)
    pa, pb = _paired_plot_result(() -> plot_result(a; view=view),
                                 () -> plot_result(b; view=view))
    @test typeof(pa) === typeof(pb)
    for f in fieldnames(typeof(pa))
        @test _deep_equal(getfield(pa, f), getfield(pb, f))
    end
    pa
end

@testset "RSER-05 nowcast serialization (#778)" begin
    @testset "registry" begin
        for name in _RSER05
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
        @test !any(v == "pending RSER-05" for v in values(_MEM._SERIALIZATION_EXCLUDED))
    end

    Y = _rser05_data()
    nM, nQ = 4, 1
    m_dfm = nowcast_dfm(Y, nM, nQ; r=1, p=1, max_iter=10, thresh=1e-2)

    @testset "NowcastDFM" begin
        @test _from_serializable_is_generic(NowcastDFM)
        args = Any[getfield(m_dfm, i) for i in 1:nfields(m_dfm)]
        @test _MEM._infer_float_param(args) === Float64
        payload = _MEM._to_serializable(m_dfm)
        @test !haskey(payload, "__struct__")
        m2 = _assert_roundtrip(m_dfm)
        _assert_consumers(m_dfm, m2)
        nr = nowcast(m_dfm)
        nr2 = nowcast(m2)
        @test nr2.model isa NowcastDFM
        @test _deep_equal(nr.nowcast, nr2.nowcast)
        @test _deep_equal(nr.forecast, nr2.forecast)
        fc = forecast(m_dfm, 4)
        fc2 = forecast(m2, 4)
        @test _deep_equal(fc.values, fc2.values)
        let path = joinpath(mktempdir(), "nowcast_dfm.jld2")
            save_model(m_dfm, path)
            m3 = load_model(path)
            @test m3 isa NowcastDFM{Float64}
            _assert_report_equal(m_dfm, m3)
            @test _deep_equal(nowcast(m_dfm).nowcast, nowcast(m3).nowcast)
            @test _deep_equal(forecast(m_dfm, 3).values, forecast(m3, 3).values)
        end
    end

    @testset "NowcastBVAR" begin
        Yb = randn(MersenneTwister(7781), 60, 4)
        Yb[55:60, 3:4] .= NaN
        m = nowcast_bvar(Yb, 2, 2; lags=2, max_iter=10)
        @test _from_serializable_is_generic(NowcastBVAR)
        m2 = _assert_roundtrip(m)
        _assert_consumers(m, m2)
        @test isnan(m2.theta_cross)  # conjugate prior
        @test m2.prior === :conjugate
        nr = nowcast(m)
        nr2 = nowcast(m2)
        @test nr2.model isa NowcastBVAR
        @test !(nr2.model isa AbstractDict)
        @test _deep_equal(nr.nowcast, nr2.nowcast)
        @test _deep_equal(forecast(m, 3).values, forecast(m2, 3).values)
    end

    @testset "NowcastBridge nested Vector{Vector{T}}" begin
        Yb = _rser05_data(T_obs=90, nM=3, nQ=2, seed=7782)
        m = nowcast_bridge(Yb, 3, 2; lagM=1, lagQ=0, lagY=1)
        @test _from_serializable_is_generic(NowcastBridge)
        args = Any[getfield(m, i) for i in 1:nfields(m)]
        @test _MEM._infer_float_param(args) === Float64
        @test m.coefficients isa Vector{Vector{Float64}}
        m2 = _assert_roundtrip(m)
        _assert_consumers(m, m2)
        @test m2.coefficients isa Vector{Vector{Float64}}
        @test eltype(m2.coefficients) === Vector{Float64}
        nr = nowcast(m)
        nr2 = nowcast(m2)
        @test nr2.model isa NowcastBridge
        @test !(nr2.model isa AbstractDict)
        @test _deep_equal(nr.nowcast, nr2.nowcast)
    end

    @testset "NowcastResult nested AbstractNowcastModel" begin
        nr = nowcast(m_dfm)
        @test _from_serializable_is_generic(NowcastResult)
        payload = _MEM._to_serializable(nr)
        @test payload["model"] isa AbstractDict
        @test payload["model"]["__struct__"] == "NowcastDFM"
        nr2 = _assert_roundtrip(nr)
        @test nr2.model isa NowcastDFM{Float64}
        @test !(nr2.model isa AbstractDict)
        _assert_consumers(nr, nr2)
        for v in (:default, :heatmap, :contributions)
            _assert_plot_view_equal(nr, nr2, v)
        end
        let path = joinpath(mktempdir(), "nowcast_result.jld2")
            save_model(nr, path)
            nr3 = load_model(path)
            @test nr3 isa NowcastResult{Float64}
            @test nr3.model isa NowcastDFM{Float64}
            _assert_report_equal(nr, nr3)
            _assert_plot_equal(nr, nr3)
            for v in (:default, :heatmap, :contributions)
                _assert_plot_view_equal(nr, nr3, v)
            end
        end
    end

    @testset "NowcastNews + nowcast_news(old2, new2)" begin
        X_new = Y
        X_old = copy(Y)
        X_old[58:60, 1:2] .= NaN
        groups = [1, 1, 2, 2, 3]
        gnames = ["Industry", "Retail", "GDP"]
        m_old = nowcast_dfm(X_old, nM, nQ; r=1, p=1, max_iter=10, thresh=1e-2)
        news = nowcast_news(X_new, X_old, m_old, 58; target_var=5,
                            groups=groups, group_names=gnames)
        @test _from_serializable_is_generic(NowcastNews)
        news2 = _assert_roundtrip(news)
        _assert_consumers(news, news2)
        @test news2.group_names == gnames
        @test _deep_equal(news2.group_impacts, news.group_impacts)
        for v in (:releases, :groups, :individual)
            _assert_plot_view_equal(news, news2, v)
        end

        m_old2 = _roundtrip(m_old)
        news_rt = nowcast_news(X_new, X_old, m_old2, 58; target_var=5,
                               groups=groups, group_names=gnames)
        @test _deep_equal(news, news_rt)

        m_new = m_dfm
        let dir = mktempdir()
            p_old = joinpath(dir, "old.jld2")
            p_new = joinpath(dir, "new.jld2")
            save_model(m_old, p_old)
            save_model(m_new, p_new)
            old2 = load_model(p_old)
            new2 = load_model(p_new)
            @test old2 isa NowcastDFM{Float64}
            @test new2 isa NowcastDFM{Float64}
            news_disk = nowcast_news(new2.data, old2.data, old2, 58; target_var=5,
                                     groups=groups, group_names=gnames)
            @test _deep_equal(news, news_disk)
            @test _deep_equal(news_disk.group_impacts, news.group_impacts)
        end
    end

    @testset "NowcastForecast Union{Vector,Matrix}" begin
        fc_mat = forecast(m_dfm, 4)
        @test _from_serializable_is_generic(NowcastForecast)
        @test fc_mat.values isa Matrix{Float64}
        fc_mat2 = _assert_roundtrip(fc_mat)
        _assert_consumers(fc_mat, fc_mat2)
        @test fc_mat2.values isa Matrix{Float64}
        @test fc_mat2.target_var === nothing

        fc_vec = forecast(m_dfm, 4; target_var=5)
        @test fc_vec.values isa Vector{Float64}
        fc_vec2 = _assert_roundtrip(fc_vec)
        _assert_consumers(fc_vec, fc_vec2)
        @test fc_vec2.values isa Vector{Float64}
        @test fc_vec2.target_var == 5
    end
end
