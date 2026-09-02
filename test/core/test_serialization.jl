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
        _assert_consumers(model, v2)
        @test v2.Y == model.Y && v2.B == model.B && v2.U == model.U
        @test v2.Sigma == model.Sigma && v2.p == model.p
        @test v2.aic == model.aic && v2.bic == model.bic && v2.hqic == model.hqic
        @test v2.varnames == model.varnames

        post = estimate_bvar(Y, 2; n_draws=50, seed=7)
        b2 = _roundtrip(post)
        @test b2 isa BVARPosterior
        _assert_consumers(post, b2)
        @test b2.B_draws == post.B_draws && b2.Sigma_draws == post.Sigma_draws
        @test b2.n_draws == post.n_draws && b2.data == post.data
        @test b2.prior == post.prior && b2.sampler == post.sampler
        @test b2.manifest isa ReproManifest && b2.manifest.seed == 7

        X = hcat(ones(100), randn(MersenneTwister(2), 100, 2))
        yv = X * [1.0, 0.5, -0.3] .+ 0.1 .* randn(MersenneTwister(3), 100)
        reg = estimate_reg(yv, X)
        r2 = _roundtrip(reg)
        @test r2 isa RegModel
        _assert_consumers(reg, r2)
        @test r2.beta == reg.beta && r2.vcov_mat == reg.vcov_mat
        @test r2.residuals == reg.residuals && r2.r2 == reg.r2
        @test r2.method == reg.method && r2.cov_type == reg.cov_type
        @test r2.weights === reg.weights    # nothing survives as nothing

        yb = Float64.((X * [0.0, 1.5, -1.5] .+ 0.3 .* randn(MersenneTwister(4), 100)) .> 0)
        logit = estimate_logit(yb, X)
        l2 = _roundtrip(logit)
        @test l2 isa LogitModel
        _assert_consumers(logit, l2)
        @test l2.beta == logit.beta && l2.vcov_mat == logit.vcov_mat
        @test l2.converged == logit.converged && l2.iterations == logit.iterations

        probit = estimate_probit(yb, X)
        pr2 = _roundtrip(probit)
        @test pr2 isa ProbitModel
        _assert_consumers(probit, pr2)
        @test pr2.beta == probit.beta && pr2.loglik == probit.loglik

        lp = estimate_lp(Y, 1, 6)
        lp2 = _roundtrip(lp)
        @test lp2 isa LPModel
        _assert_consumers(lp, lp2)
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
        m2 = _cover(tsd)
        @test m2 isa TimeSeriesData && m2.frequency == _MEM.Quarterly   # enum survives
        @test m2.vardesc == tsd.vardesc                                 # Dict survives

        dfp = DataFrame(g=repeat(1:10, inner=8), t=repeat(1:8, outer=10),
                        y=randn(MersenneTwister(2), 80), x=randn(MersenneTwister(3), 80))
        _cover(xtset(dfp, :g, :t))

        dfc = DataFrame(y=randn(MersenneTwister(4), 60), x1=randn(MersenneTwister(5), 60))
        _cover(CrossSectionData(Matrix(dfc); varnames=["y", "x1"]))

        io = load_example(:wiot)          # nested IOMetaData + Dict{String,IOExtension}
        io2 = _cover(io)
        @test io2 isa IOData && io2.meta isa IOMetaData
        _assert_roundtrip(io.meta)
    end

    @testset "cointegration / VECM" begin
        Yci = cumsum(randn(MersenneTwister(3), 150, 2); dims=1)
        _cover(estimate_vecm(Yci, 2; rank=1))   # nested JohansenResult + Vector{Matrix}

        xci = cumsum(randn(MersenneTwister(4), 120)); yci = 2 .* xci .+ randn(MersenneTwister(41), 120)
        _cover(estimate_cointreg(yci, xci; method=:fmols, trend=:const))

        dfCI = DataFrame(g=repeat(1:8, inner=30), t=repeat(1:30, outer=8))
        xp = Float64[]; yp = Float64[]
        for gg in 1:8
            xx = cumsum(randn(MersenneTwister(100 + gg), 30))
            append!(xp, xx); append!(yp, 1.5 .* xx .+ randn(MersenneTwister(200 + gg), 30))
        end
        dfCI.y = yp; dfCI.x = xp
        _cover(estimate_xtcointreg(xtset(dfCI, :g, :t), :y, :x; method=:fmols))
    end

    @testset "volatility" begin
        yv = randn(MersenneTwister(7), 400)
        _cover(estimate_arch(yv, 1))
        _cover(estimate_garch(yv, 1, 1))
        _cover(estimate_egarch(yv, 1, 1))
        _cover(estimate_gjr_garch(yv, 1, 1))
        _cover(estimate_aparch(yv, 1, 1; fix_delta=2.0, fix_gamma=0.0))
        _cover(estimate_cgarch(yv))
        # n=300/truncation=50 fits a valid FI(E)GARCH in ~0.1s; n=400/truncation=100 sent the
        # long-memory MLE into a ~36s optimizer thrash (450× slower) — the single dominant cost
        # in this file. Serialization coverage only needs a converged model, not a specific d, so
        # use the fast config (FIGARCH correctness/truncation is exercised in test_volatility.jl).
        rl = randn(MersenneTwister(38), 300)
        _cover(estimate_figarch(rl; truncation=50))
        _cover(estimate_fiegarch(rl; truncation=50))
        # GARCH-MIDAS needs > K+1 low-freq blocks (⌈n/m_freq⌉ > 13 ⇒ n ≥ 308); its own 400-obs
        # series (already fast) stays, decoupled from the shrunk FI(E)GARCH series above.
        _cover(estimate_garch_midas(randn(MersenneTwister(38), 400),
                                               randn(MersenneTwister(39), 400); K=12, m_freq=22))
        _cover(estimate_dcc(randn(MersenneTwister(40), 250, 2)))
        _cover(estimate_sv(yv[1:150]; n_samples=20, burnin=10))
    end

    @testset "factor / FAVAR" begin
        X = randn(MersenneTwister(9), 150, 8)
        _cover(estimate_factors(X, 2))
        _cover(estimate_dynamic_factors(X, 2, 1))
        _cover(estimate_gdfm(X, 2))
        _cover(estimate_favar(X, [1, 2], 2, 2))            # nested FactorModel
        _cover(estimate_structural_dfm(X, 2; p=1, H=10))   # nested GDFM + VARModel
    end

    @testset "ARIMA / ARDL / nonlinear / MIDAS / state space" begin
        ya = randn(MersenneTwister(11), 200)
        _cover(estimate_ar(ya, 2))
        _cover(estimate_ma(ya, 1))
        _cover(estimate_arma(ya, 1, 1))
        _cover(estimate_arima(ya, 1, 0, 1))
        _cover(estimate_arfima(ya, 1, 0; method=:css))
        xa = randn(MersenneTwister(41), 200)
        _cover(estimate_ardl(ya, reshape(xa, :, 1); p=1, q=1, case=3))
        _cover(estimate_nardl(ya, reshape(xa, :, 1); p=1, q=1))
        Xthr = hcat(ones(199), ya[1:199])
        _cover(estimate_threshold(ya[2:end], Xthr, randn(MersenneTwister(42), 199); linearity=false))
        _cover(estimate_midas(randn(MersenneTwister(43), 60), randn(MersenneTwister(44), 180);
                                         m=3, K=6, weights=:umidas, p_ar=0))

        # PMG (pooled mean group) panel ARDL
        NGp, TTp = 12, 25
        idv = repeat(1:NGp, inner=TTp); tmv = repeat(1:TTp, outer=NGp)
        xv = Float64[]; yv = Float64[]
        for gg in 1:NGp
            xx = cumsum(randn(MersenneTwister(300 + gg), TTp))
            append!(xv, xx); append!(yv, 0.8 .* xx .+ randn(MersenneTwister(400 + gg), TTp))
        end
        _cover(estimate_pmg(yv, reshape(xv, :, 1), idv, tmv;
                                       p=1, q=1, method=:pmg, xnames=["x"]))

        # StateSpaceModel: the `builder` closure is intentionally NOT serialized
        # (compiled functions don't round-trip); it reloads as `nothing`.
        build = θ -> (Z=reshape([1.0], 1, 1), H=reshape([exp(θ[3])], 1, 1),
                      T=reshape([tanh(θ[1])], 1, 1), Q=reshape([exp(θ[2])], 1, 1))
        yss = cumsum(randn(MersenneTwister(23), 80))
        ssm = estimate_statespace(build, [0.3, 0.0, 0.0], yss)
        ssm2 = _cover(ssm; skip=[:builder])
        @test ssm.builder isa Function      # original carried a builder…
        @test ssm2.builder === nothing       # …reload drops it
        _assert_report_equal(ssm, ssm2)
    end

    @testset "discrete / limited-dependent choice" begin
        Xo = randn(MersenneTwister(13), 200, 2)
        yo = rand(MersenneTwister(14), 1:3, 200)
        _cover(estimate_ologit(yo, Xo; varnames=["x1", "x2"]))
        _cover(estimate_oprobit(yo, Xo; varnames=["x1", "x2"]))
        Xm = hcat(ones(200), randn(MersenneTwister(15), 200, 2))
        ym = rand(MersenneTwister(16), 1:3, 200)
        _cover(estimate_mlogit(ym, Xm; varnames=["c", "x1", "x2"]))
    end

    @testset "local-projection variants" begin
        Ylp = randn(MersenneTwister(17), 150, 3)
        _cover(estimate_lp_iv(Ylp, 1, randn(MersenneTwister(18), 150, 1), 6;
                                         lags=2, cov_type=:newey_west))
        _cover(estimate_smooth_lp(Ylp, 1, 6; lambda=1.0, lags=2))   # nested BSplineBasis
        _cover(estimate_state_lp(Ylp, 1, randn(MersenneTwister(19), 150), 6;
                                            gamma=1.5, threshold=0.0, lags=2))  # nested StateTransition
        _cover(estimate_propensity_lp(Ylp, rand(MersenneTwister(20), Bool, 150),
                                                 randn(MersenneTwister(24), 150, 2), 5; lags=2))
    end

    @testset "systems / GMM" begin
        y1 = randn(MersenneTwister(30), 60); X1 = hcat(ones(60), randn(MersenneTwister(31), 60, 2))
        y2 = randn(MersenneTwister(32), 60); X2 = hcat(ones(60), randn(MersenneTwister(33), 60, 2))
        _cover(estimate_sur([(y1, X1, ["c", "v", "k"]), (y2, X2, ["c", "v", "k"])]))

        gdata = randn(MersenneTwister(21), 200, 1)
        mfn = (θ, d) -> hcat(d[:, 1] .- θ[1], (d[:, 1] .- θ[1]).^2 .- θ[2])
        _cover(estimate_gmm(mfn, [0.0, 1.0], gdata; weighting=:identity))   # nested GMMWeighting

        sim_ar1 = (θ, Tp, burn; rng=Random.default_rng()) -> begin
            n = Tp + burn; x = zeros(n)
            for t in 2:n; x[t] = θ[1] * x[t-1] + randn(rng); end
            reshape(x[burn+1:end], Tp, 1)
        end
        sdata = reshape(cumsum(randn(MersenneTwister(35), 200)) .* 0.1, 200, 1)
        _cover(estimate_smm(sim_ar1,
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
        pv = estimate_pvar(pdP, 1)
        _cover(pv)
        # v1 payloads written before boot_* / manifest fields still reconstruct
        let c = _MEM._build_container(pv)
            @test c["format_version"] == SERIALIZATION_FORMAT_VERSION == 1
            for k in ("boot_irf", "boot_lower", "boot_upper", "boot_draws", "manifest")
                delete!(c["payload"], k)
            end
            old = _MEM._reconstruct_from_container(c)
            @test old isa PVARModel
            @test old.boot_irf === nothing && old.boot_lower === nothing
            @test old.boot_upper === nothing && old.boot_draws === nothing
            @test old.manifest === nothing
            @test old.Phi == pv.Phi
        end
        _cover(estimate_xtreg(pdP, :y, [:x1, :x2]; model=:fe))
        # HDFE fit (T272, #371): carries a NamedTuple `hdfe` field
        m_hdfe = estimate_xtreg(pdP, :y, [:x1, :x2]; absorb=[:entity, :time])
        _cover(m_hdfe)
        let pth = joinpath(mktempdir(), "hdfe.jld2")
            save_model(m_hdfe, pth)
            back = load_model(pth)
            @test back.hdfe.absorb == [:entity, :time]
            @test back.hdfe.n_absorbed == m_hdfe.hdfe.n_absorbed
            @test back.hdfe.n_levels == m_hdfe.hdfe.n_levels
            @test back.hdfe.converged
            @test dof_residual(back) == dof_residual(m_hdfe)
        end
        _cover(estimate_xtiv(pdP, :y, [:x1], [:xen]; instruments=[:z1, :z2], model=:fe))
        _cover(estimate_xtlogit(pdP, :yb, [:x1, :x2]))
        _cover(estimate_xtprobit(pdP, :yb, [:x1, :x2]))
    end

    @testset "DSER-14 save_model compress= shrinks a VARModel and reloads" begin
        Y = randn(MersenneTwister(772), 400, 4)
        m = estimate_var(Y, 3)
        mktempdir() do d
            p_raw = joinpath(d, "var.jld2")
            p_z = joinpath(d, "var_z.jld2")
            save_model(m, p_raw)
            save_model(m, p_z; compress=true)
            @test filesize(p_z) < filesize(p_raw)
            m_z = load_model(p_z)
            @test m_z isa VARModel
            @test _deep_equal(m_z.B, m.B) && _deep_equal(m_z.Sigma, m.Sigma)
            @test _deep_equal(m_z.Y, m.Y)
            m_raw = load_model(p_raw)
            @test _deep_equal(m_raw.B, m.B)
        end
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

    @testset "DSGE-family: HA results are registered (DSER-07)" begin
        for name in ("HASteadyState", "HADSGESolution", "KrusellSmithSolution",
                     "WinberryFamily", "DenHaanAccuracy", "HAGridDiagnostics",
                     "HAGrid", "IncomeProcess", "HouseholdSystem", "IndividualProblem")
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
        end
        @test haskey(_MEM._SERIALIZABLE_TYPES, "DSGESolution")
        @test haskey(_MEM._SERIALIZABLE_TYPES, "DSGEEstimation")
        @test haskey(_MEM._SERIALIZABLE_TYPES, "BayesianDSGE")
        @test haskey(_MEM._SERIALIZABLE_TYPES, "DSGEPrior")
    end

    @testset "DSGE-family: SSJ blocks are registered (DSER-08)" begin
        for name in ("SSJModel", "SimpleBlock", "HetBlock", "MitBlock",
                     "SSJGEJacobian", "SSJImpulseResponse")
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
        end
    end

    @testset "DSGE-family: DCEGM / firms / intermediary (DSER-10)" begin
        for name in ("DCEGMProblem", "DCEGMSolution", "DCEGMEquilibrium",
                     "DCEGMTransition", "FirmSystem", "KhanThomasSteadyState",
                     "KhanThomasTransition", "IntermediarySystem",
                     "IntermediaryPE", "IntermediarySteadyState",
                     "IntermediaryTransition")
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
        end
    end

    @testset "DSGE-family: OLG / CT (DSER-09)" begin
        for name in ("BlanchardOLG", "BlanchardOLGSteadyState", "BlanchardOLGSolution",
                     "LifeCycleOLG", "LifeCycleSystem", "LifeCycleSteadyState",
                     "LifeCycleTransition", "CTPoissonIncome", "CTAiyagari",
                     "CTSteadyState", "CTTransition", "CTTwoAsset",
                     "CTTwoAssetSolution", "CTTwoAssetGE", "CTTwoAssetTransition",
                     "ContinuousHouseholdSystem")
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
        end
    end

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
    # Completeness flip (RSER-14 / #787): no pending reasons remain.
    @test !any(contains(v, "pending") for v in values(_MEM._SERIALIZATION_EXCLUDED))
    for name in ("HallinLiskaResult", "BaiNgQResult", "AmengualWatsonResult",
                 "IdentifiabilityTestResult")
        @test haskey(_MEM._SERIALIZABLE_TYPES, name)
        @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
    end

    # nested-array T inference
    @test _MEM._infer_float_param([[[1.0, 2.0], [3.0]]]) === Float64

    ut = deser_ser_roundtrip(1:3)          # helper below
    @test ut isa UnitRange{Int} && ut == 1:3
    vt = deser_ser_roundtrip([(1.0, 2.0)])
    @test vt isa Vector && vt[1] isa Tuple

    # LPIVARBand.sets :: Matrix{Vector{Tuple{T,T}}} — RSER-01 narrowing (#784)
    sets = Matrix{Vector{Tuple{Float64,Float64}}}(undef, 2, 1)
    sets[1, 1] = [(1.0, 2.0)]
    sets[2, 1] = [(3.0, 4.0), (5.0, 6.0)]
    sets2 = deser_ser_roundtrip(sets)
    @test sets2 isa Matrix{Vector{Tuple{Float64,Float64}}}
    @test eltype(sets2) === Vector{Tuple{Float64,Float64}}
    @test eltype(eltype(sets2)) === Tuple{Float64,Float64}
    @test sets2 == sets

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
            _assert_consumers(m, m2)
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
        _assert_consumers(svar, s2)
        @test s2 isa SVARModel
        @test s2.pattern isa SVARPattern
        @test s2.identification isa IdentificationStatus
        @test s2.varnames == svar.varnames

        Yc = cumsum(randn(MersenneTwister(754), 80, 2); dims=1)
        svec = identify_svec(estimate_vecm(Yc, 1; rank=1))
        v2 = _assert_roundtrip(svec)
        _assert_consumers(svec, v2)
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

# =============================================================================
# RSER-12 / #785 — bundles, note=, model_info
# =============================================================================

@testset "RSER-12 bundles + model_info (#785)" begin
    Y = randn(MersenneTwister(785), 80, 2)
    m = estimate_var(Y, 2)
    ir = irf(m, 4)
    fc = forecast(m, 4; ci_method=:none)

    @testset "named Dict bundle round-trips with per-entry equality" begin
        p = joinpath(mktempdir(), "bundle.jld2")
        @test save_model(Dict("var" => m, "irf" => ir, "fc" => fc), p;
                         note="session 1") == p
        @test isfile(p)
        b = load_model(p)
        @test b isa Dict{String,Any}
        @test Set(keys(b)) == Set(["var", "irf", "fc"])
        @test b["var"] isa VARModel
        @test b["irf"] isa ImpulseResponse
        @test b["fc"] isa VARForecast
        @test _deep_equal(b["var"], m)
        @test _deep_equal(b["irf"], ir)
        @test _deep_equal(b["fc"], fc)
        _assert_consumers(m, b["var"])
        _assert_consumers(ir, b["irf"])
        _assert_consumers(fc, b["fc"])

        c = _MEM._read_model_container(p)
        @test c["format_version"] == SERIALIZATION_FORMAT_VERSION
        @test c["bundle"] === true
        @test c["note"] == "session 1"
        @test c["entries"] isa AbstractDict
        @test Set(keys(c["entries"])) == Set(["var", "irf", "fc"])
        @test !haskey(c, "type")          # issue shape: "bundle"=>true, not type="__bundle__"
        @test !haskey(c, "payload")
        for k in ("var", "irf", "fc")
            @test haskey(c["entries"][k], "manifest")
            @test haskey(c["entries"][k], "type")
            @test haskey(c["entries"][k], "payload")
        end
        @test c["entries"]["var"]["type"] == "VARModel"
        @test c["entries"]["irf"]["type"] == "ImpulseResponse"
        @test c["entries"]["fc"]["type"] == "VARForecast"
    end

    @testset "vector bundle is keyed 1, 2, …" begin
        p = joinpath(mktempdir(), "vec.jld2")
        save_model([m, ir], p)
        b = load_model(p)
        @test b isa Dict{String,Any}
        @test Set(keys(b)) == Set(["1", "2"])
        @test _deep_equal(b["1"], m)
        @test _deep_equal(b["2"], ir)
    end

    @testset "note= on a single-object file; old files without note still load" begin
        p = joinpath(mktempdir(), "noted.jld2")
        save_model(m, p; note="vintage 2020")
        @test _deep_equal(load_model(p), m)
        info = model_info(p)
        @test info["note"] == "vintage 2020"
        @test info["type"] == "VARModel"
        @test info["bundle"] === false
        @test !haskey(info, "payload")
        @test info["format_version"] == SERIALIZATION_FORMAT_VERSION
        @test !isempty(info["package_version"])
        @test !isempty(info["julia_version"])
        @test !isempty(info["created"])

        # pre-RSER-12 container: no "note" key
        oldp = joinpath(mktempdir(), "old.jld2")
        _MEM._write_model_container(oldp, _MEM._build_container(m))
        @test _deep_equal(load_model(oldp), m)
        old_info = model_info(oldp)
        @test old_info["note"] == ""
        @test old_info["bundle"] === false
        @test old_info["type"] == "VARModel"
    end

    @testset "model_info returns the header without reconstructing a corrupt payload" begin
        p = joinpath(mktempdir(), "corrupt.jld2")
        save_model(m, p; note="keep me")
        c = _MEM._read_model_container(p)
        c["payload"] = "not-a-dict"
        _MEM._write_model_container(p, c)
        info = model_info(p)
        @test info["type"] == "VARModel"
        @test info["note"] == "keep me"
        @test info["bundle"] === false
        @test !haskey(info, "payload")
        @test_throws SerializationError load_model(p)

        bp = joinpath(mktempdir(), "corrupt-bundle.jld2")
        save_model(Dict("var" => m, "irf" => ir), bp; note="bundle note")
        bc = _MEM._read_model_container(bp)
        bc["entries"]["var"]["payload"] = "not-a-dict"
        _MEM._write_model_container(bp, bc)
        binfo = model_info(bp)
        @test binfo["bundle"] === true
        @test binfo["note"] == "bundle note"
        @test !haskey(binfo, "payload")
        @test binfo["entries"] isa AbstractDict
        @test binfo["entries"]["var"]["type"] == "VARModel"
        @test binfo["entries"]["irf"]["type"] == "ImpulseResponse"
        @test !haskey(binfo["entries"]["var"], "payload")
        @test !haskey(binfo["entries"]["irf"], "payload")
        @test_throws SerializationError load_model(bp)
    end

    @testset "unregistered object in a bundle raises before writing" begin
        dir = mktempdir()
        p = joinpath(dir, "bad.jld2")
        err = try
            save_model(Dict("var" => m, "bad" => 3.14), p)
            nothing
        catch e
            e
        end
        @test err isa SerializationError
        @test occursin("bundle key 'bad'", err.msg)
        @test !isfile(p)

        p2 = joinpath(dir, "badvec.jld2")
        err2 = try
            save_model([m, 3.14], p2)
            nothing
        catch e
            e
        end
        @test err2 isa SerializationError
        @test occursin("bundle key '2'", err2.msg)
        @test !isfile(p2)
    end

    @testset "model_info missing file; compress= still round-trips" begin
        @test_throws SerializationError model_info(joinpath(mktempdir(), "nope.jld2"))
        p = joinpath(mktempdir(), "z.jld2")
        save_model(Dict("var" => m), p; compress=true, note="")
        b = load_model(p)
        @test _deep_equal(b["var"], m)
        save_model(m, joinpath(mktempdir(), "z2.jld2"); compress=true)
    end
end

# =============================================================================
# RSER-14 / #787 — v1 fixtures, report/plot coverage, leftover types
# =============================================================================

const _V1_FIXTURE_DIR = joinpath(@__DIR__, "..", "fixtures", "serialization", "v1")

function _v1_report_text(x)
    applicable(report, IOBuffer(), x) ? sprint(report, x) : sprint(show, x)
end

@testset "RSER-14 committed v1 fixtures (#787)" begin
    @testset "dsge_rbc / ha_ks (DSER-12)" begin
        sol = load_model(joinpath(_V1_FIXTURE_DIR, "dsge_rbc.jld2"))
        @test sol isa DSGESolution
        @test sol.G1[1, 1] ≈ 0.9652763987036223 atol=1e-8
        @test occursin("DSGE", _v1_report_text(sol)) || occursin("G1", _v1_report_text(sol)) ||
              occursin("solution", lowercase(_v1_report_text(sol)))
        hh = load_model(joinpath(_V1_FIXTURE_DIR, "ha_ks.jld2"))
        @test hh isa HouseholdSystem
        @test hh.grid.n_points == [20]
        @test occursin("Household", _v1_report_text(hh)) ||
              occursin("huggett", lowercase(_v1_report_text(hh)))
    end

    @testset "var_irf bundle" begin
        p = joinpath(_V1_FIXTURE_DIR, "var_irf.jld2")
        info = model_info(p)
        @test info["bundle"] === true
        @test info["note"] == "v1 VAR+IRF"
        @test info["format_version"] == SERIALIZATION_FORMAT_VERSION == 1
        b = load_model(p)
        @test b isa Dict{String,Any}
        m = b["var"]; ir = b["irf"]
        @test m isa VARModel
        @test ir isa ImpulseResponse
        @test m.B[1, 1] ≈ 0.07081226845682394 atol=1e-12
        @test m.aic ≈ -0.15805202465183987 atol=1e-12
        @test ir.values[1, 1, 1] ≈ 0.8799475022032175 atol=1e-12
        @test ir.horizon == 4
        @test occursin("VAR(2)", _v1_report_text(m))
        @test occursin("Impulse Response", _v1_report_text(ir))
        _assert_report_equal(m, load_model(p)["var"])
        _assert_plot_equal(ir, load_model(p)["irf"])
    end

    @testset "bvar posterior" begin
        post = load_model(joinpath(_V1_FIXTURE_DIR, "bvar.jld2"))
        @test post isa BVARPosterior
        @test post.n_draws == 30
        @test post.manifest isa ReproManifest
        @test post.manifest.seed == 787
        @test post.B_draws[1, 1, 1] ≈ -0.03873919122810009 atol=1e-12
        @test occursin("BVAR(2)", _v1_report_text(post))
        _assert_report_equal(post, load_model(joinpath(_V1_FIXTURE_DIR, "bvar.jld2")))
        _assert_plot_equal(post, load_model(joinpath(_V1_FIXTURE_DIR, "bvar.jld2")))
    end

    @testset "arima" begin
        ar = load_model(joinpath(_V1_FIXTURE_DIR, "arima.jld2"))
        @test ar isa ARIMAModel
        @test ar.phi[1] ≈ 0.6674653302811457 atol=1e-12
        @test ar.theta[1] ≈ -0.42494294002274846 atol=1e-12
        @test ar.sigma2 ≈ 0.8179601585527572 atol=1e-12
        @test occursin("ARIMA(1,0,1)", _v1_report_text(ar))
        _assert_report_equal(ar, load_model(joinpath(_V1_FIXTURE_DIR, "arima.jld2")))
        _assert_plot_equal(ar, load_model(joinpath(_V1_FIXTURE_DIR, "arima.jld2")))
    end

    @testset "garch" begin
        g = load_model(joinpath(_V1_FIXTURE_DIR, "garch.jld2"))
        @test g isa GARCHModel
        @test g.omega ≈ 0.03726519928972168 atol=1e-12
        @test g.alpha[1] ≈ 0.016408871657496087 atol=1e-12
        @test g.beta[1] ≈ 0.9474806738106885 atol=1e-12
        @test occursin("GARCH(1,1)", _v1_report_text(g))
        _assert_report_equal(g, load_model(joinpath(_V1_FIXTURE_DIR, "garch.jld2")))
        _assert_plot_equal(g, load_model(joinpath(_V1_FIXTURE_DIR, "garch.jld2")))
    end

    @testset "factor" begin
        fm = load_model(joinpath(_V1_FIXTURE_DIR, "factor.jld2"))
        @test fm isa FactorModel
        @test fm.r == 2
        @test fm.loadings[1, 1] ≈ 0.2571180804510769 atol=1e-12
        @test fm.factors[1, 1] ≈ 1.1023609840188204 atol=1e-12
        @test occursin("Factor", _v1_report_text(fm))
        _assert_report_equal(fm, load_model(joinpath(_V1_FIXTURE_DIR, "factor.jld2")))
        _assert_plot_equal(fm, load_model(joinpath(_V1_FIXTURE_DIR, "factor.jld2")))
    end

    @testset "nowcast DFM" begin
        dfm = load_model(joinpath(_V1_FIXTURE_DIR, "nowcast_dfm.jld2"))
        @test dfm isa NowcastDFM
        @test dfm.r == 1 && dfm.nM == 4 && dfm.nQ == 1
        @test dfm.C[1, 1] ≈ 0.5361319853159928 atol=1e-12
        @test dfm.loglik ≈ -367.1564464003978 atol=1e-8
        @test nowcast(dfm).nowcast ≈ -0.19577163358544722 atol=1e-12
        @test occursin("Factor Model", _v1_report_text(dfm))
        _assert_report_equal(dfm, load_model(joinpath(_V1_FIXTURE_DIR, "nowcast_dfm.jld2")))
    end

    @testset "DiD event study" begin
        es = load_model(joinpath(_V1_FIXTURE_DIR, "did_event_study.jld2"))
        @test es isa EventStudyLP
        @test es.horizon == 3
        @test es.B[1][1, 1] ≈ 0.040550268021155765 atol=1e-12
        @test occursin("Event", _v1_report_text(es)) || occursin("event", lowercase(_v1_report_text(es)))
        _assert_report_equal(es, load_model(joinpath(_V1_FIXTURE_DIR, "did_event_study.jld2")))
        _assert_plot_equal(es, load_model(joinpath(_V1_FIXTURE_DIR, "did_event_study.jld2")))
    end

    @testset "OPP result" begin
        o = load_model(joinpath(_V1_FIXTURE_DIR, "opp.jld2"))
        @test o isa OPPResult
        @test o.H == 6
        @test o.origin == "2021Q2"
        @test o.delta[1] ≈ -0.2023713160761598 atol=1e-12
        @test o.loss_opp ≈ 7.788922361626405 atol=1e-12
        @test occursin("OPP", _v1_report_text(o)) || occursin("loss", lowercase(_v1_report_text(o)))
        _assert_report_equal(o, load_model(joinpath(_V1_FIXTURE_DIR, "opp.jld2")))
        _assert_plot_equal(o, load_model(joinpath(_V1_FIXTURE_DIR, "opp.jld2")))
    end

    @testset "teststat bundle" begin
        p = joinpath(_V1_FIXTURE_DIR, "teststat.jld2")
        info = model_info(p)
        @test info["bundle"] === true
        @test info["note"] == "v1 teststat"
        ts = load_model(p)
        adf = ts["adf"]; kpss = ts["kpss"]
        @test adf isa ADFResult
        @test kpss isa KPSSResult
        @test adf.statistic ≈ -4.713534278830027 atol=1e-12
        @test adf.pvalue ≈ 7.936190822027194e-5 atol=1e-14
        @test kpss.statistic ≈ 0.17142972566878253 atol=1e-12
        @test occursin("Dickey-Fuller", _v1_report_text(adf))
        @test occursin("KPSS", _v1_report_text(kpss))
        _assert_report_equal(adf, load_model(p)["adf"])
        _assert_report_equal(kpss, load_model(p)["kpss"])
    end
end

@testset "RSER-14 report/plot helper coverage (#787)" begin
    helper_src = read(joinpath(@__DIR__, "..", "serialization_helpers.jl"), String)
    @test occursin("function _assert_report_equal", helper_src)
    @test occursin("function _assert_plot_equal", helper_src)
    @test occursin("function _assert_consumers", helper_src)
    @test occursin("function _cover", helper_src)
    # Round-trip is structural only; consumers own report/plot (no double-report).
    @test !occursin("isempty(skip) && _assert_report_equal(m, m2)", helper_src)
    @test occursin("applicable(plot_result, a) && _assert_plot_equal", helper_src)

    plot_names = _registered_dispatch_names(plot_result)
    report_names = _registered_dispatch_names(report)
    @test length(plot_names) >= 50
    @test length(report_names) >= 50
    dser = _dser_coverage_skip()
    report_cov, plot_cov = _scan_ser_helper_coverage()
    miss_r = sort(collect(setdiff(report_names, report_cov, dser)))
    miss_p = sort(collect(setdiff(plot_names, plot_cov, dser)))
    @test miss_r == String[]
    @test miss_p == String[]

    # Runtime: a registered type with both dispatches round-trips with helpers.
    Y = randn(MersenneTwister(787), 60, 2)
    m = estimate_var(Y, 1)
    ir = irf(m, 4)
    @test string(nameof(typeof(ir))) in plot_names
    @test string(nameof(typeof(m))) in report_names
    ir2 = _roundtrip(ir)
    m2 = _roundtrip(m)
    _assert_report_equal(m, m2)
    _assert_plot_equal(ir, ir2)
end

@testset "RSER-14 leftover identifiability result (#787)" begin
    r = IdentifiabilityTestResult(:label_stability, 0.8, NaN, true,
                                  Dict{Symbol,Any}(:n => 10, :fallback => :label_stability))
    @test _from_serializable_is_generic(IdentifiabilityTestResult)
    r2 = _assert_roundtrip(r)
    @test r2.test_name === :label_stability
    @test r2.statistic == 0.8
    @test isnan(r2.pvalue)
    @test r2.identified
    @test r2.details[:n] == 10
    _assert_report_equal(r, r2)
    let path = joinpath(mktempdir(), "ident.jld2")
        save_model(r, path)
        r3 = load_model(path)
        @test r3 isa IdentifiabilityTestResult
        @test isnan(r3.pvalue)
        _assert_report_equal(r, r3)
    end
end

