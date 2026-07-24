# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using JLD2                # exercises the weak-dependency disk backend
using Random
using LinearAlgebra
using DataFrames
using Statistics

const _MEM = MacroEconometricModels

# Pure in-memory round-trip through the versioned container (no disk / no backend).
_roundtrip(m) = _MEM._reconstruct_from_container(_MEM._build_container(m))

# Recursive, NaN-aware structural equality over public fields — recurses into
# MacroEconometricModels structs, arrays, dicts, and tuples so a round-tripped
# model can be compared field-by-field even when a field is a nested struct.
function _deep_equal(a, b)
    a === nothing && return b === nothing
    a isa Missing && return b isa Missing
    if a isa Number
        return (isnan(a) && b isa Number && isnan(b)) || isequal(a, b)
    end
    (a isa AbstractString || a isa Symbol || a isa Enum || a isa Bool) && return isequal(a, b)
    if a isa AbstractArray
        b isa AbstractArray || return false
        size(a) == size(b) || return false
        return all(_deep_equal(a[i], b[i]) for i in eachindex(a))
    end
    if a isa AbstractDict
        b isa AbstractDict || return false
        Set(keys(a)) == Set(keys(b)) || return false
        return all(_deep_equal(a[k], b[k]) for k in keys(a))
    end
    a isa Tuple && return length(a) == length(b) && all(_deep_equal(a[i], b[i]) for i in eachindex(a))
    if isstructtype(typeof(a)) && parentmodule(typeof(a)) === _MEM
        typeof(a).name === typeof(b).name || return false
        return all(_deep_equal(getfield(a, f), getfield(b, f)) for f in fieldnames(typeof(a)))
    end
    return isequal(a, b)
end

# Round-trip `m` through the container and assert every public field (minus any
# in `skip`, e.g. an intentionally-dropped closure) survives structurally intact.
function _assert_roundtrip(m; skip::Vector{Symbol}=Symbol[])
    m2 = _roundtrip(m)
    @test typeof(m2).name === typeof(m).name
    for f in fieldnames(typeof(m))
        f in skip && continue
        @test _deep_equal(getfield(m, f), getfield(m2, f))
    end
    return m2
end

@testset "Versioned serialization (T248/#347)" begin

    @testset "container round-trip reconstructs public fields exactly — all types" begin
        Y = randn(MersenneTwister(1), 120, 2)

        model = estimate_var(Y, 2)
        v2 = _roundtrip(model)
        @test v2 isa VARModel
        @test v2.Y == model.Y && v2.B == model.B && v2.U == model.U
        @test v2.Sigma == model.Sigma && v2.p == model.p
        @test v2.aic == model.aic && v2.bic == model.bic && v2.hqic == model.hqic
        @test v2.varnames == model.varnames

        post = estimate_bvar(Y, 2; n_draws=50, seed=7)
        b2 = _roundtrip(post)
        @test b2 isa BVARPosterior
        @test b2.B_draws == post.B_draws && b2.Sigma_draws == post.Sigma_draws
        @test b2.n_draws == post.n_draws && b2.data == post.data
        @test b2.prior == post.prior && b2.sampler == post.sampler
        @test b2.manifest isa ReproManifest && b2.manifest.seed == 7

        X = hcat(ones(100), randn(MersenneTwister(2), 100, 2))
        yv = X * [1.0, 0.5, -0.3] .+ 0.1 .* randn(MersenneTwister(3), 100)
        reg = estimate_reg(yv, X)
        r2 = _roundtrip(reg)
        @test r2 isa RegModel
        @test r2.beta == reg.beta && r2.vcov_mat == reg.vcov_mat
        @test r2.residuals == reg.residuals && r2.r2 == reg.r2
        @test r2.method == reg.method && r2.cov_type == reg.cov_type
        @test r2.weights === reg.weights    # nothing survives as nothing

        yb = Float64.((X * [0.0, 1.5, -1.5] .+ 0.3 .* randn(MersenneTwister(4), 100)) .> 0)
        logit = estimate_logit(yb, X)
        l2 = _roundtrip(logit)
        @test l2 isa LogitModel
        @test l2.beta == logit.beta && l2.vcov_mat == logit.vcov_mat
        @test l2.converged == logit.converged && l2.iterations == logit.iterations

        probit = estimate_probit(yb, X)
        pr2 = _roundtrip(probit)
        @test pr2 isa ProbitModel
        @test pr2.beta == probit.beta && pr2.loglik == probit.loglik

        lp = estimate_lp(Y, 1, 6)
        lp2 = _roundtrip(lp)
        @test lp2 isa LPModel
        @test lp2.B == lp.B && lp2.residuals == lp.residuals && lp2.vcov == lp.vcov
        @test lp2.horizon == lp.horizon && lp2.lags == lp.lags
        @test lp2.cov_estimator isa typeof(lp.cov_estimator)
    end

    @testset "save_model / load_model disk round-trip via JLD2" begin
        Y = randn(MersenneTwister(5), 120, 3)

        model = estimate_var(Y, 2)
        path = joinpath(mktempdir(), "var.jld2")
        @test save_model(model, path) == path
        @test isfile(path)
        m2 = load_model(path)
        @test m2 isa VARModel
        @test m2.Y == model.Y && m2.B == model.B && m2.Sigma == model.Sigma
        @test m2.aic == model.aic && m2.varnames == model.varnames

        post = estimate_bvar(Y, 2; n_draws=40, seed=99)
        pp = joinpath(mktempdir(), "bvar.jld2")
        save_model(post, pp)
        p2 = load_model(pp)
        @test p2.B_draws == post.B_draws
        @test p2.manifest isa ReproManifest && p2.manifest.seed == 99   # manifest persisted

        # A reloaded BVAR still reproduces from its persisted seed
        @test reproduce(p2).matched === true
    end

    @testset "container metadata header" begin
        Y = randn(MersenneTwister(6), 80, 2)
        c = _MEM._build_container(estimate_var(Y, 2))
        @test c["format_version"] == SERIALIZATION_FORMAT_VERSION
        @test c["type"] == "VARModel"
        @test !isempty(c["package_version"])
        @test !isempty(c["julia_version"])
        @test haskey(c, "payload") && c["payload"] isa AbstractDict
    end

    @testset "top-level manifest travels with the container" begin
        Y = randn(MersenneTwister(7), 80, 2)
        post = estimate_bvar(Y, 2; n_draws=30, seed=11)
        c = _MEM._build_container(post)
        @test c["manifest"] isa AbstractDict
        @test c["manifest"]["seed"] == 11
        # a deterministic result has no manifest
        cv = _MEM._build_container(estimate_var(Y, 2))
        @test cv["manifest"] === nothing
    end

    @testset "unknown format_version and type raise a typed, informative error" begin
        Y = randn(MersenneTwister(8), 80, 2)
        c = _MEM._build_container(estimate_var(Y, 2))

        bad_ver = copy(c); bad_ver["format_version"] = 999
        err = try
            _MEM._reconstruct_from_container(bad_ver); nothing
        catch e
            e
        end
        @test err isa SerializationError
        @test occursin("999", err.msg)
        @test occursin(string(SERIALIZATION_FORMAT_VERSION), err.msg)

        bad_type = copy(c); bad_type["type"] = "NopeModel"
        @test_throws SerializationError _MEM._reconstruct_from_container(bad_type)

        no_ver = copy(c); delete!(no_ver, "format_version")
        @test_throws SerializationError _MEM._reconstruct_from_container(no_ver)
    end

    @testset "unsupported save target and missing file raise SerializationError" begin
        @test_throws SerializationError _MEM._build_container(3.14)
        @test_throws SerializationError load_model(joinpath(mktempdir(), "does_not_exist.jld2"))
    end
end

@testset "Full model & data-container coverage (#505)" begin

    @testset "data containers" begin
        Y = randn(MersenneTwister(1), 120, 3)
        tsd = TimeSeriesData(Y; varnames=["a", "b", "c"], frequency=_MEM.Quarterly,
                             vardesc=Dict("a" => "alpha"))
        m2 = _assert_roundtrip(tsd)
        @test m2 isa TimeSeriesData && m2.frequency == _MEM.Quarterly   # enum survives
        @test m2.vardesc == tsd.vardesc                                 # Dict survives

        dfp = DataFrame(g=repeat(1:10, inner=8), t=repeat(1:8, outer=10),
                        y=randn(MersenneTwister(2), 80), x=randn(MersenneTwister(3), 80))
        _assert_roundtrip(xtset(dfp, :g, :t))

        dfc = DataFrame(y=randn(MersenneTwister(4), 60), x1=randn(MersenneTwister(5), 60))
        _assert_roundtrip(CrossSectionData(Matrix(dfc); varnames=["y", "x1"]))

        io = load_example(:wiot)          # nested IOMetaData + Dict{String,IOExtension}
        io2 = _assert_roundtrip(io)
        @test io2 isa IOData && io2.meta isa IOMetaData
        _assert_roundtrip(io.meta)
    end

    @testset "cointegration / VECM" begin
        Yci = cumsum(randn(MersenneTwister(3), 150, 2); dims=1)
        _assert_roundtrip(estimate_vecm(Yci, 2; rank=1))   # nested JohansenResult + Vector{Matrix}

        xci = cumsum(randn(MersenneTwister(4), 120)); yci = 2 .* xci .+ randn(MersenneTwister(41), 120)
        _assert_roundtrip(estimate_cointreg(yci, xci; method=:fmols, trend=:const))

        dfCI = DataFrame(g=repeat(1:8, inner=30), t=repeat(1:30, outer=8))
        xp = Float64[]; yp = Float64[]
        for gg in 1:8
            xx = cumsum(randn(MersenneTwister(100 + gg), 30))
            append!(xp, xx); append!(yp, 1.5 .* xx .+ randn(MersenneTwister(200 + gg), 30))
        end
        dfCI.y = yp; dfCI.x = xp
        _assert_roundtrip(estimate_xtcointreg(xtset(dfCI, :g, :t), :y, :x; method=:fmols))
    end

    @testset "volatility" begin
        yv = randn(MersenneTwister(7), 400)
        _assert_roundtrip(estimate_arch(yv, 1))
        _assert_roundtrip(estimate_garch(yv, 1, 1))
        _assert_roundtrip(estimate_egarch(yv, 1, 1))
        _assert_roundtrip(estimate_gjr_garch(yv, 1, 1))
        _assert_roundtrip(estimate_aparch(yv, 1, 1; fix_delta=2.0, fix_gamma=0.0))
        _assert_roundtrip(estimate_cgarch(yv))
        # n=300/truncation=50 fits a valid FI(E)GARCH in ~0.1s; n=400/truncation=100 sent the
        # long-memory MLE into a ~36s optimizer thrash (450× slower) — the single dominant cost
        # in this file. Serialization coverage only needs a converged model, not a specific d, so
        # use the fast config (FIGARCH correctness/truncation is exercised in test_volatility.jl).
        rl = randn(MersenneTwister(38), 300)
        _assert_roundtrip(estimate_figarch(rl; truncation=50))
        _assert_roundtrip(estimate_fiegarch(rl; truncation=50))
        # GARCH-MIDAS needs > K+1 low-freq blocks (⌈n/m_freq⌉ > 13 ⇒ n ≥ 308); its own 400-obs
        # series (already fast) stays, decoupled from the shrunk FI(E)GARCH series above.
        _assert_roundtrip(estimate_garch_midas(randn(MersenneTwister(38), 400),
                                               randn(MersenneTwister(39), 400); K=12, m_freq=22))
        _assert_roundtrip(estimate_dcc(randn(MersenneTwister(40), 250, 2)))
        _assert_roundtrip(estimate_sv(yv[1:150]; n_samples=20, burnin=10))
    end

    @testset "factor / FAVAR" begin
        X = randn(MersenneTwister(9), 150, 8)
        _assert_roundtrip(estimate_factors(X, 2))
        _assert_roundtrip(estimate_dynamic_factors(X, 2, 1))
        _assert_roundtrip(estimate_gdfm(X, 2))
        _assert_roundtrip(estimate_favar(X, [1, 2], 2, 2))            # nested FactorModel
        _assert_roundtrip(estimate_structural_dfm(X, 2; p=1, H=10))   # nested GDFM + VARModel
    end

    @testset "ARIMA / ARDL / nonlinear / MIDAS / state space" begin
        ya = randn(MersenneTwister(11), 200)
        _assert_roundtrip(estimate_ar(ya, 2))
        _assert_roundtrip(estimate_ma(ya, 1))
        _assert_roundtrip(estimate_arma(ya, 1, 1))
        _assert_roundtrip(estimate_arima(ya, 1, 0, 1))
        _assert_roundtrip(estimate_arfima(ya, 1, 0; method=:css))
        xa = randn(MersenneTwister(41), 200)
        _assert_roundtrip(estimate_ardl(ya, reshape(xa, :, 1); p=1, q=1, case=3))
        _assert_roundtrip(estimate_nardl(ya, reshape(xa, :, 1); p=1, q=1))
        Xthr = hcat(ones(199), ya[1:199])
        _assert_roundtrip(estimate_threshold(ya[2:end], Xthr, randn(MersenneTwister(42), 199); linearity=false))
        _assert_roundtrip(estimate_midas(randn(MersenneTwister(43), 60), randn(MersenneTwister(44), 180);
                                         m=3, K=6, weights=:umidas, p_ar=0))

        # PMG (pooled mean group) panel ARDL
        NGp, TTp = 12, 25
        idv = repeat(1:NGp, inner=TTp); tmv = repeat(1:TTp, outer=NGp)
        xv = Float64[]; yv = Float64[]
        for gg in 1:NGp
            xx = cumsum(randn(MersenneTwister(300 + gg), TTp))
            append!(xv, xx); append!(yv, 0.8 .* xx .+ randn(MersenneTwister(400 + gg), TTp))
        end
        _assert_roundtrip(estimate_pmg(yv, reshape(xv, :, 1), idv, tmv;
                                       p=1, q=1, method=:pmg, xnames=["x"]))

        # StateSpaceModel: the `builder` closure is intentionally NOT serialized
        # (compiled functions don't round-trip); it reloads as `nothing`.
        build = θ -> (Z=reshape([1.0], 1, 1), H=reshape([exp(θ[3])], 1, 1),
                      T=reshape([tanh(θ[1])], 1, 1), Q=reshape([exp(θ[2])], 1, 1))
        yss = cumsum(randn(MersenneTwister(23), 80))
        ssm = estimate_statespace(build, [0.3, 0.0, 0.0], yss)
        ssm2 = _assert_roundtrip(ssm; skip=[:builder])
        @test ssm.builder isa Function      # original carried a builder…
        @test ssm2.builder === nothing       # …reload drops it
    end

    @testset "discrete / limited-dependent choice" begin
        Xo = randn(MersenneTwister(13), 200, 2)
        yo = rand(MersenneTwister(14), 1:3, 200)
        _assert_roundtrip(estimate_ologit(yo, Xo; varnames=["x1", "x2"]))
        _assert_roundtrip(estimate_oprobit(yo, Xo; varnames=["x1", "x2"]))
        Xm = hcat(ones(200), randn(MersenneTwister(15), 200, 2))
        ym = rand(MersenneTwister(16), 1:3, 200)
        _assert_roundtrip(estimate_mlogit(ym, Xm; varnames=["c", "x1", "x2"]))
    end

    @testset "local-projection variants" begin
        Ylp = randn(MersenneTwister(17), 150, 3)
        _assert_roundtrip(estimate_lp_iv(Ylp, 1, randn(MersenneTwister(18), 150, 1), 6;
                                         lags=2, cov_type=:newey_west))
        _assert_roundtrip(estimate_smooth_lp(Ylp, 1, 6; lambda=1.0, lags=2))   # nested BSplineBasis
        _assert_roundtrip(estimate_state_lp(Ylp, 1, randn(MersenneTwister(19), 150), 6;
                                            gamma=1.5, threshold=0.0, lags=2))  # nested StateTransition
        _assert_roundtrip(estimate_propensity_lp(Ylp, rand(MersenneTwister(20), Bool, 150),
                                                 randn(MersenneTwister(24), 150, 2), 5; lags=2))
    end

    @testset "systems / GMM" begin
        y1 = randn(MersenneTwister(30), 60); X1 = hcat(ones(60), randn(MersenneTwister(31), 60, 2))
        y2 = randn(MersenneTwister(32), 60); X2 = hcat(ones(60), randn(MersenneTwister(33), 60, 2))
        _assert_roundtrip(estimate_sur([(y1, X1, ["c", "v", "k"]), (y2, X2, ["c", "v", "k"])]))

        gdata = randn(MersenneTwister(21), 200, 1)
        mfn = (θ, d) -> hcat(d[:, 1] .- θ[1], (d[:, 1] .- θ[1]).^2 .- θ[2])
        _assert_roundtrip(estimate_gmm(mfn, [0.0, 1.0], gdata; weighting=:identity))   # nested GMMWeighting

        sim_ar1 = (θ, Tp, burn; rng=Random.default_rng()) -> begin
            n = Tp + burn; x = zeros(n)
            for t in 2:n; x[t] = θ[1] * x[t-1] + randn(rng); end
            reshape(x[burn+1:end], Tp, 1)
        end
        sdata = reshape(cumsum(randn(MersenneTwister(35), 200)) .* 0.1, 200, 1)
        _assert_roundtrip(estimate_smm(sim_ar1,
            d -> [Statistics.mean(d[:, 1]), Statistics.var(d[:, 1])],
            [0.3], sdata; weighting=:identity, sim_ratio=2))
    end

    @testset "panel / PVAR" begin
        NG, TT = 20, 10; N = NG * TT
        dfP = DataFrame(g=repeat(1:NG, inner=TT), t=repeat(1:TT, outer=NG),
                        y=randn(MersenneTwister(50), N), x1=randn(MersenneTwister(51), N),
                        x2=randn(MersenneTwister(52), N), xen=randn(MersenneTwister(53), N),
                        z1=randn(MersenneTwister(54), N), z2=randn(MersenneTwister(55), N),
                        yb=Float64.(rand(MersenneTwister(56), N) .> 0.5))
        pdP = xtset(dfP, :g, :t)
        _assert_roundtrip(estimate_pvar(pdP, 1))
        _assert_roundtrip(estimate_xtreg(pdP, :y, [:x1, :x2]; model=:fe))
        _assert_roundtrip(estimate_xtiv(pdP, :y, [:x1], [:xen]; instruments=[:z1, :z2], model=:fe))
        _assert_roundtrip(estimate_xtlogit(pdP, :yb, [:x1, :x2]))
        _assert_roundtrip(estimate_xtprobit(pdP, :yb, [:x1, :x2]))
    end

    @testset "disk round-trip via JLD2 for a non-VAR model + data container" begin
        Y = randn(MersenneTwister(70), 120, 2)
        vecm = estimate_vecm(cumsum(Y; dims=1), 2; rank=1)
        pth = joinpath(mktempdir(), "vecm.jld2")
        @test save_model(vecm, pth) == pth
        v2 = load_model(pth)
        @test v2 isa VECMModel && _deep_equal(v2.Pi, vecm.Pi) && _deep_equal(v2.beta, vecm.beta)

        tsd = TimeSeriesData(Y; varnames=["a", "b"], frequency=_MEM.Monthly)
        tp = joinpath(mktempdir(), "tsd.jld2")
        save_model(tsd, tp)
        t2 = load_model(tp)
        @test t2 isa TimeSeriesData && t2.frequency == _MEM.Monthly && _deep_equal(t2.data, tsd.data)
    end

    @testset "every registered type is a recognized save/load target" begin
        # Registry ↔ dispatch: every tag maps to a distinct concrete type name.
        for (tag, T) in _MEM._SERIALIZABLE_TYPES
            @test tag == string(nameof(T))
        end
        @test length(_MEM._SERIALIZABLE_TYPES) >= 50
    end

    @testset "DSGE-family carve-out: compiled-equation models are not yet supported" begin
        # These transitively hold a DSGESpec whose residual_fns are compiled
        # closures; they are deliberately absent from the registry (follow-up).
        for name in ("DSGESolution", "BayesianDSGE", "HADSGESolution",
                     "KrusellSmithSolution", "DSGEEstimation")
            @test !haskey(_MEM._SERIALIZABLE_TYPES, name)
        end
    end
end
