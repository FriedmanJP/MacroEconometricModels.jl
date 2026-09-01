# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
using JLD2                # exercises the JLD2 disk backend

# In-memory leaf round-trip (no container / no disk).
deser_ser_roundtrip(x) = _MEM._deser_field(_MEM._ser_field(x))

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
        # HDFE fit (T272, #371): carries a NamedTuple `hdfe` field
        m_hdfe = estimate_xtreg(pdP, :y, [:x1, :x2]; absorb=[:entity, :time])
        _assert_roundtrip(m_hdfe)
        let pth = joinpath(mktempdir(), "hdfe.jld2")
            save_model(m_hdfe, pth)
            back = load_model(pth)
            @test back.hdfe.absorb == [:entity, :time]
            @test back.hdfe.n_absorbed == m_hdfe.hdfe.n_absorbed
            @test back.hdfe.n_levels == m_hdfe.hdfe.n_levels
            @test back.hdfe.converged
            @test dof_residual(back) == dof_residual(m_hdfe)
        end
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

    # DSGE-family carve-out retired: completeness + `_SERIALIZATION_EXCLUDED`
    # pending reasons (RSER-01) own that list (`DSGESolution`, `BayesianDSGE`,
    # `HADSGESolution`, `KrusellSmithSolution`, `DSGEEstimation`, …).
end

@testset "DSER-01 leaf codecs" begin
    ser, deser = _MEM._ser_field, _MEM._deser_field

    ex = :(a + log(b) ^ 2)
    @test deser(ser(ex)) == Base.remove_linenums!(ex)
    qn = Expr(:call, :+, QuoteNode(:x), 1)
    @test deser(ser(qn)) == qn
    nt = (midpoints=(points=:a, max=1.5), n=3)
    @test deser(ser(nt)) == nt
    pr = :K => identity
    pr2 = deser(ser(pr))
    @test pr2 isa Pair && pr2.first === :K

    @test deser(ser(_MEM._ks_budget)) === _MEM._ks_budget
    @test ser(x -> x) === nothing   # anonymous; owning type decides (DSER-06)

    using SparseArrays
    S = sprand(MersenneTwister(759), 8, 8, 0.3)
    S2 = deser(ser(S))
    @test S2 isa SparseMatrixCSC && Array(S2) == Array(S)

    F = lu(randn(MersenneTwister(760), 4, 4))
    @test ser(F) === nothing

    d = _MEM._build_container(estimate_var(randn(MersenneTwister(1), 40, 2), 1))
    _MEM._assert_plain_payload(d)
end

@testset "RSER-01 completeness + generic hardening" begin
    exported_names = String[]
    for n in names(MacroEconometricModels)
        T = getfield(MacroEconometricModels, n)
        T isa Type || continue
        while T isa UnionAll
            T = T.body
        end
        T isa DataType || continue
        isstructtype(T) || continue
        T <: Exception && continue
        startswith(string(n), "_") && continue
        push!(exported_names, string(n))
    end
    for name in exported_names
        @test haskey(_MEM._SERIALIZABLE_TYPES, name) ||
              haskey(_MEM._SERIALIZATION_EXCLUDED, name)
    end

    # nested-array T inference
    @test _MEM._infer_float_param([[[1.0, 2.0], [3.0]]]) === Float64

    ut = deser_ser_roundtrip(1:3)          # helper below
    @test ut isa UnitRange{Int} && ut == 1:3
    vt = deser_ser_roundtrip([(1.0, 2.0)])
    @test vt isa Vector && vt[1] isa Tuple

    # Keyword-constructor detector: every registered type reconstructs from
    # positional field values, or has an explicit `_from_serializable` override.
    Y = randn(MersenneTwister(774), 40, 2)
    samples = Any[
        estimate_var(Y, 1),
        estimate_factors(randn(MersenneTwister(775), 50, 6), 1),
        TimeSeriesData(Y; varnames=["a", "b"]),
    ]
    for m in samples
        T = typeof(m)
        Tw = Base.typename(T).wrapper
        if _from_serializable_is_generic(Tw)
            args = Any[getfield(m, i) for i in 1:nfields(m)]
            @test _MEM._generic_construct(Tw, args) isa T
        else
            @test which(_MEM._from_serializable, Tuple{Type{Tw}, AbstractDict, Int}).sig isa DataType
        end
    end
    for (name, T) in _MEM._SERIALIZABLE_TYPES
        m = which(_MEM._from_serializable, Tuple{Type{T}, AbstractDict, Int})
        @test (m.sig isa UnionAll) || (m.sig isa DataType)
    end
end

# =============================================================================
# SID-24 / #753 — identification result serialization
# =============================================================================

# Typed SID-14 restrictions (no closures). Function-valued `identify_sign` checks
# are not serializable; those results store `restrictions === nothing`.
_sid24_restrictions(n::Int=2) = SVARRestrictions(n;
    zeros=[ZeroRestriction(2, 1, 0)],
    signs=[SignRestriction(1, 1, 0, 1)])

function _sid24_dummy_objects()
    T = Float64
    n = 2
    B0 = T[1.2 0.3; 0.1 0.9]
    Q = Matrix{T}(I, n, n)
    shocks = randn(MersenneTwister(7531), 12, n)
    se = fill(T(0.05), n, n)
    vcov = Matrix{T}(I, 3, 3)
    snames = ["Shock 1", "Shock 2"]
    vnames = ["y1", "y2"]
    restr = _sid24_restrictions(n)
    idst = IdentificationStatus(:exact, [1, 1], [1, 1], 0)
    I2 = Matrix{T}(I, n, n)
    irf4 = randn(MersenneTwister(7532), 2, 3, n, n)
    irf3 = randn(MersenneTwister(7533), 3, n, n)

    ica = ICASVARResult{T}(B0, inv(B0), Q, shocks, :fastica, true, 10, T(0.1), snames)
    ml = NonGaussianMLResult{T}(B0, Q, shocks, :student_t, T(-10), T(-12),
                                 Dict{Symbol,Any}(:nu => T[5.0, 6.0]),
                                 Matrix{T}(I, 4, 4), se, true, 20, T(22), T(24), snames)
    gmm = NonGaussianGMMResult{T}(B0, Q, T[0.1], Matrix{T}(I, 1, 1), se,
                                   T(1.2), T(0.3), :coskewness, :two_step,
                                   shocks, vnames, snames)
    ms = MarkovSwitchingSVARResult{T}(B0, Q, [I2, 2 .* I2],
                                       [T[1.0, 1.0], T[2.0, 0.5]],
                                       fill(T(0.5), 12, 2), T[0.9 0.1; 0.1 0.9],
                                       T(-10), true, 5, 2, se, vcov, T(0.8),
                                       shocks, snames)
    garch = GARCHSVARResult{T}(B0, Q, T[0.1 0.1 0.8; 0.1 0.1 0.8],
                                ones(T, 12, n), shocks, T(-10), true, 8,
                                se, vcov, snames)
    st = SmoothTransitionSVARResult{T}(B0, Q, [I2, 2 .* I2],
                                        [T[1.0, 1.0], T[2.0, 0.5]],
                                        T(1.5), T(0.0), randn(MersenneTwister(7534), 12),
                                        fill(T(0.5), 12), T(-10), true, 6,
                                        se, vcov, shocks, snames)
    ext = ExternalVolatilitySVARResult{T}(B0, Q, [I2, 2 .* I2],
                                           [T[1.0, 1.0], T[2.0, 0.5]],
                                           [[1, 2, 3], [4, 5, 6]], T(-10),
                                           se, vcov, shocks, snames)
    proxy = ProxySVARResult{T}(Q, B0, 1, T(15), T(0.5), ["z1"], vnames,
                                ["Proxy", "Unidentified 1"], true)
    maxs = MaxShareResult{T}(Q, Q[:, 1], 1, 0:20, nothing, T(0.4), T[0.8, 0.2],
                              vnames, ["Max share", "Complement"], true)
    maxs_band = MaxShareResult{T}(Q, Q[:, 1], 1, nothing, (T(0.1), T(0.5)),
                                   T(0.4), T[0.8, 0.2], vnames,
                                   ["Max share", "Complement"], true)
    arias = AriasSVARResult{T}([Q, Q], irf4, T[0.5, 0.5], T(0.4), restr,
                                T(2), T(1), vnames, 0, 0)
    uhlig = UhligSVARResult{T}(Q, irf3, T(0.1), T[0.1, 0.0], restr, true,
                                vnames, IdentificationStatus(:set, [0, 0], [0, 0], 0))
    bset = BayesianSetIdentifiedSVAR{T}([Q, Q], irf4, T[0.5, 0.5], T(2), restr,
                                         vnames, 0, 0, irf4, irf3, T[0.5, 0.5], T(1), 0)
    signs = SignIdentifiedSet{T}([Q], irf4[1:1, :, :, :], 1, 10, T(0.1),
                                  vnames, snames, T[1.0], T(1), T(1), restr)
    signs_fn = SignIdentifiedSet{T}([Q], irf4[1:1, :, :, :], 1, 10, T(0.1),
                                     vnames, snames, T[1.0], T(1), T(1), nothing)
    rb = RobustBayesResult{T}(irf3, irf3 .+ 1, irf3 .- T(0.1), irf3 .+ T(1.1),
                               irf3, irf3 .+ T(0.5), T(0.2), T(0.0), T(0.68))
    return [ica, ml, gmm, ms, garch, st, ext, proxy, maxs, maxs_band,
            arias, uhlig, bset, signs, signs_fn, rb]
end

@testset "SID-24 identification result serialization (#753)" begin
    Random.seed!(753)

    @testset "nested restriction types resolve by name" begin
        for name in ("ZeroRestriction", "SignRestriction", "SVARRestrictions",
                     "LongRunZeroRestriction", "A0ZeroRestriction",
                     "AplusZeroRestriction", "A0SignRestriction",
                     "AplusSignRestriction", "SVARPattern", "IdentificationStatus")
            @test _MEM._resolve_ser_type(name) isa Type
        end
        @test _MEM._resolve_ser_type("ZeroRestriction") === ZeroRestriction
        @test _MEM._resolve_ser_type("SignRestriction") === SignRestriction
        @test _MEM._resolve_ser_type("SVARRestrictions") === SVARRestrictions
    end

    @testset "dummy identification results round-trip fieldwise" begin
        for m in _sid24_dummy_objects()
            @test haskey(_MEM._SERIALIZABLE_TYPES, string(nameof(typeof(m))))
            m2 = _assert_roundtrip(m)
            @test typeof(m2) == typeof(m)
        end
    end

    @testset "typed nested restrictions survive" begin
        arias = _sid24_dummy_objects()[findfirst(x -> x isa AriasSVARResult,
                                                 _sid24_dummy_objects())]
        a2 = _roundtrip(arias)
        @test a2.restrictions isa SVARRestrictions
        @test a2.restrictions.zeros[1] isa ZeroRestriction
        @test a2.restrictions.zeros[1].variable == 2
        @test a2.restrictions.signs[1] isa SignRestriction
        @test a2.restrictions.signs[1].sign == 1
        uhlig = _sid24_dummy_objects()[findfirst(x -> x isa UhligSVARResult,
                                                 _sid24_dummy_objects())]
        u2 = _roundtrip(uhlig)
        @test u2.restrictions.zeros[1] isa ZeroRestriction
        @test u2.id_status.status === :set
    end

    @testset "SVARModel / SVECResult nested pattern and VECM" begin
        Y = randn(MersenneTwister(753), 80, 2)
        svar = estimate_svar(estimate_var(Y, 1), recursive_pattern(2);
                             rng=MersenneTwister(753))
        s2 = _assert_roundtrip(svar)
        @test s2 isa SVARModel
        @test s2.pattern isa SVARPattern
        @test s2.identification isa IdentificationStatus
        @test s2.varnames == svar.varnames

        Yc = cumsum(randn(MersenneTwister(754), 80, 2); dims=1)
        svec = identify_svec(estimate_vecm(Yc, 1; rank=1))
        v2 = _assert_roundtrip(svec)
        @test v2 isa SVECResult
        @test v2.vecm isa VECMModel
        @test v2.n_permanent == svec.n_permanent
    end

    @testset "disk round-trip via JLD2 for identification results" begin
        proxy = _sid24_dummy_objects()[findfirst(x -> x isa ProxySVARResult,
                                                 _sid24_dummy_objects())]
        pth = joinpath(mktempdir(), "proxy.jld2")
        @test save_model(proxy, pth) == pth
        p2 = load_model(pth)
        @test p2 isa ProxySVARResult && _deep_equal(p2.B0, proxy.B0)

        arias = _sid24_dummy_objects()[findfirst(x -> x isa AriasSVARResult,
                                                 _sid24_dummy_objects())]
        ap = joinpath(mktempdir(), "arias.jld2")
        save_model(arias, ap)
        a2 = load_model(ap)
        @test a2 isa AriasSVARResult
        @test a2.restrictions.zeros[1] isa ZeroRestriction
        @test _deep_equal(a2.weights, arias.weights)
    end
end
