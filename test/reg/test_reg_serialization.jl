# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end
using DelimitedFiles

const _RSER06 = ("RobustRegModel", "PenalizedRegModel", "HeckmanModel",
                 "TobitModel", "TruncRegModel", "PoissonModel", "NegBinModel",
                 "DispersionTest", "QuantileRegModel", "RDDResult",
                 "SelectionResult", "MarginalEffects", "MultinomialMarginalEffects",
                 "OddsRatio", "InfluenceStats", "StabilityResult",
                 "RegDiagnosticResult", "AndersonRubinTest", "AndersonRubinCI",
                 "WildClusterBootstrap", "PanelTestResult")

# True when this package defines `f` for `types` (skips StatsAPI StatisticalModel
# fallbacks that then throw MethodError on a missing primitive).
function _defined_here(f, types...)
    for meth in methods(f, Tuple{types...})
        meth.module === _MEM && return true
    end
    false
end

function _assert_predict_equal(m, m2)
    if hasfield(typeof(m), :X)
        X = getfield(m, :X)
        if _defined_here(predict, typeof(m), typeof(X))
            @test predict(m2, X) == predict(m, X)
            return
        end
    end
    _defined_here(predict, typeof(m)) && @test predict(m2) == predict(m)
    nothing
end

function _assert_statsapi(m, m2)
    Tm = typeof(m)
    _defined_here(coef, Tm) && @test coef(m2) == coef(m)
    _defined_here(vcov, Tm) && @test isequal(vcov(m2), vcov(m))
    _defined_here(stderror, Tm) && @test isequal(stderror(m2), stderror(m))
    _defined_here(residuals, Tm) && @test residuals(m2) == residuals(m)
    _defined_here(fitted, Tm) && @test isequal(fitted(m2), fitted(m))
    if _defined_here(loglikelihood, Tm)
        @test loglikelihood(m2) == loglikelihood(m)
        _defined_here(aic, Tm) && @test aic(m2) == aic(m)
    end
    _assert_predict_equal(m, m2)
    nothing
end

function _rser06_ols(n=60; seed=779)
    rng = MersenneTwister(seed)
    X = hcat(ones(n), randn(rng, n), randn(rng, n))
    y = X * [1.0, 0.6, -0.4] .+ 0.4 .* randn(rng, n)
    (y, X)
end

@testset "RSER-06 cross-section/micro serialization (#779)" begin
    @testset "registry" begin
        @test length(_RSER06) == 21
        for name in _RSER06
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
        @test !any(v == "pending RSER-06" for v in values(_MEM._SERIALIZATION_EXCLUDED))
    end

    y, X = _rser06_ols()
    vn = ["const", "x1", "x2"]
    ols = estimate_reg(y, X; varnames=vn)

    @testset "RobustRegModel" begin
        m = estimate_robust(y, X; psi=:huber, method=:m, varnames=vn)
        @test _from_serializable_is_generic(RobustRegModel)
        m2 = _assert_roundtrip(m)
        _assert_consumers(m, m2)
        _assert_statsapi(m, m2)
    end

    @testset "PenalizedRegModel" begin
        Xp = X[:, 2:3]
        m = estimate_elastic_net(y, Xp; alpha=0.5, lambda=0.1)
        @test _from_serializable_is_generic(PenalizedRegModel)
        m2 = _assert_roundtrip(m)
        _assert_consumers(m, m2)
        _assert_statsapi(m, m2)
        @test predict(m2, Xp) == predict(m, Xp)
        @test m2.cv_mse === nothing
    end

    @testset "HeckmanModel" begin
        rng = MersenneTwister(7791)
        n = 160
        z = randn(rng, n)
        x = 0.3 .* z .+ randn(rng, n)
        u = randn(rng, n)
        v = 0.6 .* u .+ sqrt(1 - 0.36) .* randn(rng, n)
        d = Float64.((0.2 .+ 0.8 .* z .+ 0.3 .* x .+ v) .> 0)
        ystar = 1.0 .+ 0.7 .* x .+ u
        yh = ifelse.(d .== 1, ystar, NaN)
        Xh = hcat(ones(n), x)
        Zh = hcat(ones(n), z, x)
        m = estimate_heckman(yh, Xh, d, Zh; method=:twostep,
                             outcome_names=["const", "x"],
                             select_names=["const", "z", "x"])
        @test _from_serializable_is_generic(HeckmanModel)
        m2 = _assert_roundtrip(m)
        _assert_consumers(m, m2)
        _assert_statsapi(m, m2)
        @test predict(m2) == predict(m)
    end

    @testset "TobitModel / TruncRegModel" begin
        n = 80
        x1 = [sin(0.3i + 0.7) for i in 1:n]
        x2 = [0.5cos(0.2i) + 0.3sin(0.5i) for i in 1:n]
        e  = [0.9sin(1.3i + 0.5) + 0.45cos(2.1i + 0.2) for i in 1:n]
        ystar = [0.4 + x1[i] - 0.7 * x2[i] + e[i] for i in 1:n]
        yt = max.(ystar, 0.0)
        Xt = hcat(ones(n), x1, x2)
        m = estimate_tobit(yt, Xt; lower=0.0, varnames=["const", "x1", "x2"])
        @test _from_serializable_is_generic(TobitModel)
        m2 = _assert_roundtrip(m)
        _assert_consumers(m, m2)
        _assert_statsapi(m, m2)
        me = marginal_effects(m)
        me2 = marginal_effects(m2)
        @test _deep_equal(me, me2)

        keep = yt .> 0
        mt = estimate_truncreg(ystar[keep], Xt[keep, :]; lower=0.0,
                               varnames=["const", "x1", "x2"])
        @test _from_serializable_is_generic(TruncRegModel)
        mt2 = _assert_roundtrip(mt)
        _assert_consumers(mt, mt2)
        _assert_statsapi(mt, mt2)
    end

    @testset "PoissonModel / NegBinModel / DispersionTest NamedTuple" begin
        d = readdlm(joinpath(@__DIR__, "data", "count_oracle.csv"), ',', Float64)
        n = size(d, 1)
        Xc = hcat(ones(n), d[:, 4], d[:, 5])
        vn_c = ["const", "x1", "x2"]
        mp = estimate_poisson(d[:, 1], Xc; varnames=vn_c, cov_type=:mle)
        @test _from_serializable_is_generic(PoissonModel)
        mp2 = _assert_roundtrip(mp)
        _assert_consumers(mp, mp2)
        _assert_statsapi(mp, mp2)
        @test predict(mp2, Xc) == predict(mp, Xc)
        @test _deep_equal(marginal_effects(mp), marginal_effects(mp2))

        mnb = estimate_nbreg(d[:, 2], Xc; varnames=vn_c)
        @test _from_serializable_is_generic(NegBinModel)
        mnb2 = _assert_roundtrip(mnb)
        _assert_consumers(mnb, mnb2)
        _assert_statsapi(mnb, mnb2)
        @test predict(mnb2, Xc) == predict(mnb, Xc)

        dt = dispersion_test(mp)
        @test _from_serializable_is_generic(DispersionTest)
        payload = _MEM._to_serializable(dt)
        @test payload["nb2"] isa AbstractDict
        @test payload["nb2"]["__namedtuple__"] === true
        @test payload["nb1"]["__namedtuple__"] === true
        args = Any[getfield(dt, i) for i in 1:nfields(dt)]
        @test _MEM._infer_float_param(args) === Float64
        dt2 = _assert_roundtrip(dt)
        _assert_consumers(dt, dt2)
        @test dt2.nb2 isa NamedTuple
        @test dt2.nb2 == dt.nb2
        @test dt2.nb1 == dt.nb1
        @test keys(dt2.nb2) === (:alpha, :se, :t_stat, :p_value)
        _MEM._assert_plain_payload(_MEM._build_container(dt))
    end

    @testset "QuantileRegModel nested Vector{Matrix{T}}" begin
        mq = estimate_qreg(y, X, [0.25, 0.5, 0.75]; se=:iid, varnames=vn)
        @test _from_serializable_is_generic(QuantileRegModel)
        @test mq.vcov_mats isa Vector{<:AbstractMatrix}
        args = Any[getfield(mq, i) for i in 1:nfields(mq)]
        @test _MEM._infer_float_param(args) === Float64
        mq2 = _assert_roundtrip(mq)
        _assert_consumers(mq, mq2)
        _assert_statsapi(mq, mq2)
        @test mq2.vcov_mats isa Vector{Matrix{Float64}}
        @test eltype(mq2.vcov_mats) === Matrix{Float64}
        @test predict(mq2, X) == predict(mq, X)
        let path = joinpath(mktempdir(), "qreg.jld2")
            save_model(mq, path)
            mq3 = load_model(path)
            @test mq3 isa QuantileRegModel{Float64}
            @test mq3.vcov_mats isa Vector{Matrix{Float64}}
            @test predict(mq3, X) == predict(mq, X)
            _assert_report_equal(mq, mq3)
        end
    end

    @testset "RDDResult Tuple CI" begin
        rng = MersenneTwister(7792)
        nrd = 400
        xr = 2 .* rand(rng, nrd) .- 1
        yr = @. 1.0 + 0.5 * xr + 0.8 * (xr >= 0) + 0.25 * randn(rng)
        rd = estimate_rdd(yr, xr; cutoff=0.0, kernel=:uniform, p=1, h=0.5, b=0.5)
        @test _from_serializable_is_generic(RDDResult)
        @test rd.ci_conventional isa Tuple
        @test rd.first_stage === nothing
        rd2 = _assert_roundtrip(rd)
        _assert_consumers(rd, rd2)
        @test rd2.ci_conventional isa NTuple{2,Float64}
        @test rd2.ci_robust isa NTuple{2,Float64}
        @test rd2.first_stage === nothing
        let path = joinpath(mktempdir(), "rdd.jld2")
            save_model(rd, path)
            rd3 = load_model(path)
            @test rd3 isa RDDResult{Float64}
            _assert_report_equal(rd, rd3)
            @test rd3.ci_robust == rd.ci_robust
        end
    end

    @testset "SelectionResult nested RegModel" begin
        rng = MersenneTwister(7793)
        n = 80
        Q = Matrix(qr(randn(rng, n, 6)).Q)[:, 1:6] .* sqrt(n)
        Xs = hcat(ones(n), Q)
        beta = zeros(7); beta[1] = 1.0; beta[[2, 4, 6]] .= 3.0
        ys = Xs * beta .+ 0.5 .* randn(rng, n)
        sel = select_variables(ys, Xs; method=:forward, criterion=:aic)
        @test _from_serializable_is_generic(SelectionResult)
        payload = _MEM._to_serializable(sel)
        @test payload["final"] isa AbstractDict
        @test payload["final"]["__struct__"] == "RegModel"
        sel2 = _assert_roundtrip(sel)
        _assert_consumers(sel, sel2)
        @test sel2.final isa RegModel{Float64}
        @test !(sel2.final isa AbstractDict)
        @test sel2.path isa Vector{<:Tuple}
        Xsel = sel.final.X
        @test predict(sel2.final, Xsel) == predict(sel.final, Xsel)
        @test vif(sel2.final) == vif(sel.final)
    end

    @testset "MarginalEffects / OddsRatio / classification_table" begin
        rng = MersenneTwister(7794)
        n = 120
        Xl = hcat(ones(n), randn(rng, n), randn(rng, n))
        yb = Float64.((Xl * [0.0, 0.6, -0.5] .+ randn(rng, n)) .> 0)
        logit = estimate_logit(yb, Xl; varnames=vn)
        me = marginal_effects(logit)
        @test _from_serializable_is_generic(MarginalEffects)
        me2 = _assert_roundtrip(me)
        _assert_consumers(me, me2)
        @test _deep_equal(marginal_effects(logit), me)

        or = odds_ratio(logit)
        @test _from_serializable_is_generic(OddsRatio)
        or2 = _assert_roundtrip(or)
        _assert_consumers(or, or2)

        logit2 = _roundtrip(logit)
        ct = classification_table(logit)
        ct2 = classification_table(logit2)
        @test ct2["accuracy"] == ct["accuracy"]
        @test ct2["confusion"] == ct["confusion"]
        @test _deep_equal(odds_ratio(logit2), or)
        @test _deep_equal(marginal_effects(logit2), me)
    end

    @testset "MultinomialMarginalEffects" begin
        rng = MersenneTwister(7795)
        n = 150
        Xm = hcat(ones(n), randn(rng, n), randn(rng, n))
        V = Xm * [0.0 0.6; 0.8 -0.5; -0.4 0.3]
        y = Vector{Int}(undef, n)
        for i in 1:n
            scores = [0.0, V[i, 1], V[i, 2]]
            scores .-= maximum(scores)
            p = exp.(scores); p ./= sum(p)
            u = rand(rng); c = 0.0; y[i] = 3
            for j in 1:3
                c += p[j]
                if u < c
                    y[i] = j; break
                end
            end
        end
        mm = estimate_mlogit(y, Xm; varnames=vn)
        mme = marginal_effects(mm)
        @test _from_serializable_is_generic(MultinomialMarginalEffects)
        mme2 = _assert_roundtrip(mme)
        _assert_consumers(mme, mme2)
        @test mme2.se isa Matrix{Float64}
    end

    @testset "InfluenceStats / StabilityResult / RegDiagnosticResult" begin
        inf = influence_stats(ols)
        @test _from_serializable_is_generic(InfluenceStats)
        inf2 = _assert_roundtrip(inf)
        _assert_consumers(inf, inf2)

        st = cusum_test(ols)
        @test _from_serializable_is_generic(StabilityResult)
        @test st.first_crossing === nothing || st.first_crossing isa Int
        st2 = _assert_roundtrip(st)
        _assert_consumers(st, st2)

        w = white_test(ols)
        @test _from_serializable_is_generic(RegDiagnosticResult)
        @test w.df isa Int
        @test w.f_stat === nothing
        w2 = _assert_roundtrip(w)
        _assert_consumers(w, w2)

        rs = reset_test(ols)
        @test rs.df isa Tuple
        rs2 = _assert_roundtrip(rs)
        _assert_consumers(rs, rs2)
        @test rs2.df isa NTuple{2,Int}

        bg = breusch_godfrey_test(ols; lags=1)
        bg2 = _assert_roundtrip(bg)
        _assert_consumers(bg, bg2)
        @test bg2.f_df isa NTuple{2,Int}
    end

    @testset "AndersonRubinTest / AndersonRubinCI Vector{Tuple}" begin
        rng = MersenneTwister(7796)
        n = 200
        z = randn(rng, n)
        vv = randn(rng, n)
        u = 0.8 .* vv .+ 0.6 .* randn(rng, n)
        xen = 0.8 .* z .+ vv
        yiv = 1.0 .* xen .+ u
        Xiv = hcat(ones(n), xen)
        Ziv = hcat(ones(n), z)
        iv = estimate_iv(yiv, Xiv, Ziv; endogenous=[2], varnames=["const", "x"], cov_type=:ols)
        ar = anderson_rubin_test(iv, 1.0; cov_type=:ols)
        @test _from_serializable_is_generic(AndersonRubinTest)
        ar2 = _assert_roundtrip(ar)
        _assert_consumers(ar, ar2)

        ci = anderson_rubin_ci(iv; n_grid=51, span=8)
        @test _from_serializable_is_generic(AndersonRubinCI)
        @test ci.intervals isa Vector{<:Tuple}
        ci2 = _assert_roundtrip(ci)
        _assert_consumers(ci, ci2)
        @test ci2.intervals isa Vector{Tuple{Float64,Float64}}
    end

    @testset "WildClusterBootstrap disk" begin
        G, n_per = 6, 8
        rng = MersenneTwister(7797)
        n = G * n_per
        cl = repeat(1:G, inner=n_per)
        xw = randn(rng, n)
        u = zeros(n)
        for g in 1:G
            a = randn(rng)
            idx = findall(==(g), cl)
            u[idx] .= sqrt(0.5) * a .+ sqrt(0.5) .* randn(rng, length(idx))
        end
        yw = 1.0 .+ 0.4 .* xw .+ u
        Xw = hcat(ones(n), xw)
        mw = estimate_reg(yw, Xw; varnames=["const", "x"])
        b = wild_cluster_bootstrap(mw, "x", 0.0; clusters=cl, n_boot=64,
                                   ci=false, rng=MersenneTwister(1))
        @test _from_serializable_is_generic(WildClusterBootstrap)
        b2 = _assert_roundtrip(b)
        _assert_consumers(b, b2)
        @test b2.t_boot == b.t_boot
        let path = joinpath(mktempdir(), "wcb.jld2")
            save_model(b, path)
            b3 = load_model(path)
            @test b3 isa WildClusterBootstrap{Float64}
            _assert_report_equal(b, b3)
            @test b3.t_boot == b.t_boot
            @test b3.n_boot == b.n_boot
        end
    end

    @testset "PanelTestResult Int and Tuple df" begin
        rng = MersenneTwister(7798)
        N_g, T_p = 20, 12
        n = N_g * T_p
        ids = repeat(1:N_g, inner=T_p)
        ts = repeat(1:T_p, N_g)
        alpha = repeat(randn(rng, N_g), inner=T_p)
        x1 = 0.5 .* alpha .+ randn(rng, n)
        x2 = randn(rng, n)
        yp = alpha .+ 2.0 .* x1 .- 1.0 .* x2 .+ 0.3 .* randn(rng, n)
        df = DataFrame(id=ids, t=ts, x1=x1, x2=x2, y=yp)
        pd = xtset(df, :id, :t)
        fe = estimate_xtreg(pd, :y, [:x1, :x2]; model=:fe)
        re = estimate_xtreg(pd, :y, [:x1, :x2]; model=:re)
        ht = hausman_test(fe, re)
        @test _from_serializable_is_generic(PanelTestResult)
        @test ht.df isa Int
        ht2 = _assert_roundtrip(ht)
        _assert_consumers(ht, ht2)
        @test ht2.df isa Int

        ft = f_test_fe(fe)
        @test ft.df isa Tuple
        ft2 = _assert_roundtrip(ft)
        _assert_consumers(ft, ft2)
        @test ft2.df isa NTuple{2,Int}
    end
end
