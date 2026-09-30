# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using MacroEconometricModels
using Test
using DataFrames
using Tables
using StatsAPI
using Random
using LinearAlgebra
using DelimitedFiles

# Tiny staggered panel for direct DiD-result construction (no estimation).
function _tidy_minipanel()
    rng = Xoshiro(7)
    n_units, n_periods = 12, 10
    treat_times = vcat(fill(4, 4), fill(7, 4), zeros(Int, 4))
    N_obs = n_units * n_periods
    data = Matrix{Float64}(undef, N_obs, 2)
    group_id = Vector{Int}(undef, N_obs)
    time_id = Vector{Int}(undef, N_obs)
    row = 1
    for i in 1:n_units, t in 1:n_periods
        data[row, 1] = randn(rng) + (treat_times[i] > 0 && t >= treat_times[i] ? 2.0 : 0.0)
        data[row, 2] = Float64(treat_times[i])
        group_id[row] = i
        time_id[row] = t
        row += 1
    end
    PanelData{Float64}(data, ["outcome", "treat_time"], Quarterly, [1, 1],
        group_id, time_id, nothing, ["unit_$i" for i in 1:n_units],
        n_units, 2, N_obs, true, ["tiny"], Dict{String,String}(), Symbol[])
end

@testset "Tables.jl integration (T247/#346)" begin

    # ── Coefficient-bearing models: DataFrame(result) ───────────────────────────
    @testset "DataFrame(RegModel) matches report inputs" begin
        rng = Xoshiro(11)
        X = randn(rng, 90, 3); y = X * [1.0, -0.5, 0.3] .+ randn(rng, 90)
        m = estimate_reg(y, X)
        df = DataFrame(m)
        @test names(df) == ["term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        @test nrow(df) == length(coef(m))
        @test df.estimate ≈ coef(m)
        @test df.std_error ≈ stderror(m)
        @test df.stat ≈ coef(m) ./ stderror(m)
        @test all(df.ci_lower .< df.ci_upper)
        # Tables source protocol.
        @test Tables.istable(m)
        @test Tables.columnaccess(typeof(m))
        @test Set(Tables.columnnames(Tables.columns(m))) ==
              Set([:term, :estimate, :std_error, :stat, :p_value, :ci_lower, :ci_upper])
        @test Tables.schema(m) !== nothing
    end

    @testset "DataFrame(LogitModel/ProbitModel)" begin
        rng = Xoshiro(12)
        X = randn(rng, 150, 2); z = X * [0.8, -0.6]
        y = Float64.(rand(rng, 150) .< 1 ./ (1 .+ exp.(-z)))
        lm = estimate_logit(y, X)
        dl = DataFrame(lm)
        @test dl.estimate ≈ coef(lm)
        @test dl.std_error ≈ stderror(lm)
        pm = estimate_probit(y, X)
        @test DataFrame(pm).estimate ≈ coef(pm)
    end

    @testset "DataFrame(MarginalEffects) drops non-finite rows" begin
        rng = Xoshiro(13)
        X = randn(rng, 150, 2); z = X * [0.8, -0.6]
        y = Float64.(rand(rng, 150) .< 1 ./ (1 .+ exp.(-z)))
        me = marginal_effects(estimate_logit(y, X))
        dme = DataFrame(me)
        keep = findall(isfinite, me.effects)
        @test nrow(dme) == length(keep)
        @test dme.estimate ≈ me.effects[keep]
        @test dme.p_value ≈ me.p_values[keep]
        @test dme.ci_lower ≈ me.ci_lower[keep]
    end

    @testset "DataFrame(OrderedModel) — two blocks" begin
        rng = Xoshiro(14)
        n = 300; X = randn(rng, n, 2); latent = X * [1.0, -0.8] .+ randn(rng, n)
        y = [v < -0.7 ? 1 : v < 0.7 ? 2 : 3 for v in latent]
        om = estimate_ologit(y, X)
        d = DataFrame(om)
        @test "block" in names(d)
        @test Set(d.block) == Set(["coef", "cutpoint"])
        @test count(==("coef"), d.block) == length(om.beta)
        @test count(==("cutpoint"), d.block) == length(om.cutpoints)
        @test d.estimate ≈ vcat(om.beta, om.cutpoints)
    end

    @testset "DataFrame(MultinomialLogitModel) — per-alternative blocks" begin
        rng = Xoshiro(15)
        n = 400; X = randn(rng, n, 2)
        # 3-category DGP.
        u2 = X * [1.0, 0.0]; u3 = X * [0.0, 1.0]
        y = map(1:n) do i
            e = -log.(-log.(rand(rng, 3)))
            argmax([0.0 + e[1], u2[i] + e[2], u3[i] + e[3]])
        end
        ml = estimate_mlogit(y, X)
        d = DataFrame(ml)
        @test "alternative" in names(d)
        @test nrow(d) == length(ml.varnames) * size(ml.beta, 2)
        @test length(unique(d.alternative)) == size(ml.beta, 2)
        @test d.estimate ≈ vec(ml.beta)
    end

    @testset "DataFrame(VARModel) — one row per (equation, term)" begin
        rng = Xoshiro(16)
        Y = randn(rng, 120, 2); vm = estimate_var(Y, 2)
        d = DataFrame(vm)
        @test "equation" in names(d)
        @test Set(d.equation) == Set(vm.varnames)
        @test nrow(d) == length(vm.varnames) * (1 + 2 * 2)   # (intercept + n*p) per equation
        # First equation's estimates equal B[:,1].
        d1 = d[d.equation .== vm.varnames[1], :]
        @test d1.estimate ≈ vm.B[:, 1]
    end

    # ── long_table for array-valued results ─────────────────────────────────────
    @testset "long_table(ImpulseResponse)" begin
        rng = Xoshiro(17)
        vm = estimate_var(randn(rng, 120, 3), 2)
        ir = irf(vm, 10; method=:cholesky)
        lt = long_table(ir)
        @test names(lt) == ["horizon", "variable", "shock", "value", "lower", "upper"]
        @test nrow(lt) == 10 * 3 * 3
        @test Set(lt.horizon) == Set(1:10)
        # Bands present when ci_type != :none, else missing.
        if ir.ci_type == :none
            @test all(ismissing, lt.lower)
        end
    end

    @testset "long_table(FEVD) and forecast" begin
        rng = Xoshiro(18)
        vm = estimate_var(randn(rng, 120, 2), 2)
        lf = long_table(fevd(vm, 8))
        @test names(lf) == ["horizon", "variable", "shock", "value"]
        @test nrow(lf) == 8 * 2 * 2
        @test all(0 .<= lf.value .<= 1 .+ 1e-8)
        lfc = long_table(forecast(vm, 6))
        @test names(lfc) == ["horizon", "variable", "value", "lower", "upper"]
        @test nrow(lfc) == 6 * 2
    end

    @testset "long_table(LPImpulseResponse) — direct construction" begin
        vals = reshape(collect(1.0:12.0), 6, 2)
        lpir = MacroEconometricModels.LPImpulseResponse{Float64}(vals, vals .- 1, vals .+ 1,
                                               fill(0.5, 6, 2), 5, ["y1", "y2"], "shock", :hac, 0.95)
        lt = long_table(lpir)
        @test names(lt) == ["horizon", "variable", "shock", "value", "se", "lower", "upper"]
        @test nrow(lt) == 6 * 2
        @test Set(lt.horizon) == Set(0:5)          # LP horizons are 0-based (impact included)
        @test all(lt.shock .== "shock")
    end

    @testset "long_table(HistoricalDecomposition) (#862)" begin
        rng = Xoshiro(21)
        vm = estimate_var(randn(rng, 60, 2), 1)
        hd = historical_decomposition(vm, 20; method=:cholesky)
        lt = long_table(hd)
        @test names(lt) == ["time", "variable", "shock", "value"]
        @test nrow(lt) == hd.T_eff * 2 * 2
        @test Set(lt.time) == Set(1:hd.T_eff)
        @test lt.value[1] ≈ hd.contributions[1, 1, 1]
        @test lt.value[end] ≈ hd.contributions[end, end, end]
        # write_csv routes HD through long_table.
        path = tempname() * ".csv"
        write_csv(hd, path)
        raw, hdr = readdlm(path, ',', header=true)
        @test vec(hdr) == ["time", "variable", "shock", "value"]

        # Bayesian: point estimate plus outer-quantile interval.
        MEM = MacroEconometricModels
        pe = reshape(collect(1.0:24.0), 4, 3, 2)
        q = cat(pe .- 1, pe, pe .+ 1; dims=4)
        bhd = MEM.BayesianHistoricalDecomposition{Float64}(
            q, pe, zeros(4, 3, 3), zeros(4, 3), zeros(4, 2), zeros(4, 3),
            4, ["y1", "y2", "y3"], ["e1", "e2"], [0.16, 0.5, 0.84], :cholesky)
        blt = long_table(bhd)
        @test names(blt) == ["time", "variable", "shock", "value", "lower", "upper"]
        @test nrow(blt) == 4 * 3 * 2
        @test blt.value ≈ vec([pe[t, v, s] for t in 1:4 for v in 1:3 for s in 1:2])
        @test blt.lower ≈ blt.value .- 1
        @test blt.upper ≈ blt.value .+ 1
    end

    @testset "long_table(BayesianFEVD) (#864)" begin
        MEM = MacroEconometricModels
        pe = reshape(collect(1.0:24.0) ./ 100, 2, 2, 6)
        q = cat(pe .- 0.01, pe, pe .+ 0.01; dims=4)
        bf = MEM.BayesianFEVD{Float64}(q, pe, 6, ["y1", "y2"], ["e1", "e2"], [0.16, 0.5, 0.84])
        lt = long_table(bf)
        @test names(lt) == ["horizon", "variable", "shock", "value", "lower", "upper"]
        @test nrow(lt) == 6 * 2 * 2
        # Same (horizon, variable, shock) keys as long_table(::FEVD).
        fevd_keys = Set(zip(lt.horizon, lt.variable, lt.shock))
        @test fevd_keys == Set((h, "y$v", "e$s") for h in 1:6 for v in 1:2 for s in 1:2)
        @test lt.value ≈ vec([pe[v, s, h] for h in 1:6 for v in 1:2 for s in 1:2])
        @test lt.lower ≈ lt.value .- 0.01
        @test lt.upper ≈ lt.value .+ 0.01
    end

    @testset "long_table(LPFEVD) (#865)" begin
        MEM = MacroEconometricModels
        raw = fill(0.25, 2, 2, 5)
        bc = reshape(collect(1.0:20.0) ./ 100, 2, 2, 5)
        se = fill(0.05, 2, 2, 5)
        lp = MEM.LPFEVD{Float64}(raw, bc, se, bc .- 0.1, bc .+ 0.1,
                                 :r2, 5, 200, 0.95, true, ["y1", "y2"], ["e1", "e2"])
        lt = long_table(lp)
        @test names(lt) == ["horizon", "variable", "shock", "value", "se", "lower", "upper"]
        @test nrow(lt) == 5 * 2 * 2
        @test Set(lt.horizon) == Set(1:5)
        # Headline value is the bias-corrected estimate, not raw proportions.
        @test lt.value ≈ vec([bc[v, s, h] for h in 1:5 for v in 1:2 for s in 1:2])
        @test all(lt.se .≈ 0.05)
        @test lt.lower ≈ lt.value .- 0.1
        @test lt.upper ≈ lt.value .+ 0.1
    end

    @testset "long_table(MidasForecast) labels the direct horizon (#867)" begin
        MEM = MacroEconometricModels
        f = MEM.MidasForecast{Float64}([1.5], [1.2], [1.8], [0.15], 4, 0.95)
        lt = long_table(f)
        @test names(lt) == ["horizon", "variable", "value", "se", "lower", "upper"]
        @test nrow(lt) == 1
        @test lt.horizon == [4]                    # NOT 1 (the pre-#867 mislabel)
        @test lt.value == [1.5]
        @test lt.se == [0.15]
        @test (lt.lower, lt.upper) == ([1.2], [1.8])
    end

    # ── TIDY coefficient families (v1.0.2, #853 series) ─────────────────────────
    @testset "DataFrame(PoissonModel/NegBinModel) (#854)" begin
        d = readdlm(joinpath(@__DIR__, "..", "reg", "data", "count_oracle.csv"), ',', Float64)
        X = hcat(ones(size(d, 1)), d[:, 4], d[:, 5])
        VN = ["const", "x1", "x2"]
        mp = estimate_poisson(d[:, 1], X; varnames=VN)
        df = DataFrame(mp)
        @test names(df) == ["term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        @test df.estimate ≈ coef(mp)
        @test df.std_error ≈ stderror(mp)
        mn = estimate_nbreg(d[:, 2], X; varnames=VN)
        dn = DataFrame(mn)
        @test names(dn) == ["block", "term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        @test dn.block == ["coef", "coef", "coef", "dispersion"]
        @test dn.term[end] == "alpha"
        @test dn.estimate[end] ≈ mn.alpha
        @test dn.std_error[end] ≈ mn.alpha_se
    end

    @testset "DataFrame(QuantileRegModel) (#855)" begin
        rng = Xoshiro(22)
        X = hcat(ones(200), randn(rng, 200, 2))
        y = X * [1.0, 0.5, -0.3] .+ randn(rng, 200)
        m = estimate_qreg(y, X, [0.25, 0.5, 0.75]; varnames=["const", "x1", "x2"])
        df = DataFrame(m)
        @test names(df) == ["tau", "term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        @test nrow(df) == 9
        @test df.tau == repeat([0.25, 0.5, 0.75], inner=3)
        @test df.estimate ≈ vec(m.beta)
        @test df.std_error ≈ vec(m.stderr)
        d05 = df[df.tau .== 0.5, :]
        @test d05.term == ["const", "x1", "x2"]
    end

    @testset "DataFrame(RDDResult) (#855)" begin
        rng = Xoshiro(23)
        running = randn(rng, 600)
        y = 0.5 .* running .+ 2.0 .* (running .>= 0) .+ randn(rng, 600)
        r = estimate_rdd(y, running; cutoff=0.0)
        df = DataFrame(r)
        @test names(df) == ["term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper", "h", "b"]
        @test df.term == ["Conventional", "Robust (bias-corrected)"]
        @test df.estimate ≈ [r.tau_conventional, r.tau_bias_corrected]
        @test df.std_error ≈ [r.se_conventional, r.se_robust]
        @test df.h == fill(r.h, 2) && df.b == fill(r.b, 2)
    end

    @testset "DataFrame(SURModel/ThreeSLSModel) (#856)" begin
        PD = load_example(:grunfeld)
        GE = group_data(PD, "General Electric")
        WH = group_data(PD, "Westinghouse")
        T = 20
        y1 = GE.data[:, 1]; X1 = hcat(ones(T), GE.data[:, 2], GE.data[:, 3])
        y2 = WH.data[:, 1]; X2 = hcat(ones(T), WH.data[:, 2], WH.data[:, 3])
        VN = ["const", "value", "capital"]
        eqs = [(y1, X1, VN), (y2, X2, VN)]
        ms = estimate_sur(eqs)
        ds = DataFrame(ms)
        @test names(ds) == ["equation", "term", "estimate", "std_error", "stat", "p_value",
            "ci_lower", "ci_upper", "nobs", "mcelroy_r2", "det_sigma", "loglik"]
        @test ds.equation == repeat(ms.eqnames, inner=3)
        @test ds[ds.equation .== ms.eqnames[1], :estimate] ≈ ms.betas[1]
        @test ds[ds.equation .== ms.eqnames[2], :estimate] ≈ ms.betas[2]
        @test all(ds.mcelroy_r2 .≈ ms.mcelroy_r2)
        @test all(ds.loglik .≈ ms.loglik)
        Z = hcat(ones(T), GE.data[:, 3], WH.data[:, 3])
        m3 = estimate_3sls(eqs, Z; eqnames=["GE", "Westinghouse"])
        d3 = DataFrame(m3)
        @test "n_instruments" in names(d3)
        @test d3[d3.equation .== "GE", :estimate] ≈ m3.betas[1]
        @test d3[d3.equation .== "Westinghouse", :estimate] ≈ m3.betas[2]
        @test all(d3.n_instruments .== 3)
    end

    @testset "DataFrame(EventStudyLP/LPDiDResult/BaconDecomposition) (#866)" begin
        MEM = MacroEconometricModels
        pd = _tidy_minipanel()
        et = [-2, -1, 0, 1, 2]
        co = [0.1, 0.0, 1.0, 1.8, 2.2]
        se = fill(0.3, 5)
        eslp = MEM.EventStudyLP{Float64}(co, se, co .- 0.6, co .+ 0.6, et, -1,
            Matrix{Float64}[], Matrix{Float64}[], Matrix{Float64}[], Int[],
            "outcome", "treat_time", 120, 12, 1, 2, 2, false, :unit, 0.95, pd)
        de = DataFrame(eslp)
        @test names(de) == ["event_time", "term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        @test de.event_time == et
        @test de.term == ["h=$e" for e in et]
        @test de.estimate ≈ co
        @test de.ci_lower ≈ co .- 0.6
        lpd = MEM.LPDiDResult{Float64}(co, se, co .- 0.6, co .+ 0.6, et, -1, fill(100, 5),
            (coef=1.9, se=0.2, ci_lower=1.5, ci_upper=2.3, nobs=300),
            (coef=0.05, se=0.15, ci_lower=-0.25, ci_upper=0.35, nobs=200),
            Matrix{Float64}[], "outcome", "treat_time", 1200, 12, :absorbing, nothing,
            false, false, 1, 0, 2, 2, :unit, 0.95, pd)
        dl = DataFrame(lpd)
        @test dl.block == vcat(fill("dynamic", 5), fill("pooled", 2))
        @test dl.term[6:7] == ["Pre-pooled", "Post-pooled"]
        @test ismissing(dl.event_time[6]) && ismissing(dl.event_time[7])
        @test collect(skipmissing(dl.event_time)) == et
        @test dl.estimate[6:7] ≈ [0.05, 1.9]
        bd = MEM.BaconDecomposition{Float64}([2.0, 1.5, 2.2], [0.5, 0.2, 0.3],
            [:earlier_vs_later, :later_vs_earlier, :treated_vs_untreated],
            [5, 8, 5], [8, 5, 0], 2.0)
        db = DataFrame(bd)
        @test names(db) == ["type", "cohort_i", "cohort_j", "estimate", "weight"]
        @test db.weight ≈ [0.5, 0.2, 0.3]
        @test db.cohort_j == [8, 5, 0]
        @test db.type[3] == "treated_vs_untreated"
    end

    @testset "Tables form for per-category marginal effects (#863)" begin
        MEM = MacroEconometricModels
        rng = Xoshiro(24)
        # Multinomial DGP: K=3 (const + 2), J=3.
        n = 400
        beta_true = [0.5 -0.3; 1.0 -0.5; -0.5 0.8]
        X = [ones(n) randn(rng, n, 2)]
        V = X * beta_true
        y = Vector{Int}(undef, n)
        for i in 1:n
            w = [1.0, exp(V[i, 1]), exp(V[i, 2])]
            p = w ./ sum(w)
            u, cum = rand(rng), 0.0
            y[i] = 3
            for j in 1:3
                cum += p[j]
                if u < cum
                    y[i] = j
                    break
                end
            end
        end
        m = estimate_mlogit(y, X; varnames=["const", "x1", "x2"])
        me = marginal_effects(m)
        @test me isa MEM.MultinomialMarginalEffects
        df = DataFrame(me)
        @test names(df) == ["variable", "category", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        @test nrow(df) == length(me.varnames) * (length(me.categories) - 1)
        @test Set(df.category) == Set(me.categories[2:end])     # base skipped
        @test df.estimate[1] ≈ me.effects[1, 2]
        # Ordered DGP: same long shape via long_table on the returned NamedTuple.
        Xo = randn(rng, 400, 2)
        xb = Xo * [1.0, -0.5]
        cuts = [0.0, 1.5]
        yo = Vector{Int}(undef, 400)
        for i in 1:400
            u = rand(rng)
            yo[i] = u < 1 / (1 + exp(-(cuts[1] - xb[i]))) ? 1 :
                    u < 1 / (1 + exp(-(cuts[2] - xb[i]))) ? 2 : 3
        end
        mo = estimate_ologit(yo, Xo; varnames=["x1", "x2"])
        nto = marginal_effects(mo)
        lt = long_table(nto)
        @test names(lt) == ["variable", "category", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        @test nrow(lt) == 2 * 3                                  # all J kept
        @test lt.estimate[1] ≈ nto.effects[1, 1]
        @test lt.std_error[1] ≈ nto.se[1, 1]
    end

    @testset "DataFrame(ForecastEvaluation/ForecastCombination) (#857)" begin
        rng = Xoshiro(25)
        T = 100
        actual = cumsum(randn(rng, T))
        f1 = actual .+ randn(rng, T)
        f2 = actual .+ 2 .* randn(rng, T)
        ev = forecast_evaluate(actual, hcat(f1, f2); model_names=["AR", "RW"])
        df = DataFrame(ev)
        @test names(df) == ["model", "ME", "MAE", "RMSE", "MAPE", "sMAPE", "MASE",
            "U1", "U2", "theil_bias", "theil_variance", "theil_covariance", "n"]
        @test df.model == ["AR", "RW"]
        @test df.n == [T, T]
        @test df.ME ≈ ev.values[:, 1]
        @test df.U2 ≈ ev.values[:, 8]
        @test df.theil_bias .+ df.theil_variance .+ df.theil_covariance ≈ ones(2)
        c = combine_forecasts(hcat(f1, f2), actual; method=:equal, model_names=["AR", "RW"])
        dc = DataFrame(c)
        @test names(dc) == ["model", "weight", "mse", "method"]
        @test dc.weight ≈ [0.5, 0.5]
        @test dc.method == ["equal", "equal"]
        @test dc.mse ≈ c.mse
    end

    @testset "DataFrame(policy counterfactuals) (#858)" begin
        MEM = MacroEconometricModels
        pc = MEM.PolicyCounterfactual{Float64}([:y], [:r],
            [[1.0, 2.0, 3.0]], [[0.1, 0.2, 0.3]], [[1.1, 2.1, 3.1]], [[0.15, 0.25, 0.35]],
            [hcat([1.0, 2.0, 3.0], [1.2, 2.2, 3.2])], nothing,
            [0.5], ["mp"], zeros(3), 0.01, nothing, true, "taylor", 3,
            [0.16, 0.84], 100, 0)
        dp = DataFrame(pc)
        @test names(dp) == ["period", "variable", "role", "baseline", "counterfactual", "lower", "upper"]
        @test nrow(dp) == 6
        @test dp[dp.role .== "outcome", :variable] == fill("y", 3)
        @test dp[dp.role .== "outcome", :counterfactual] ≈ [1.1, 2.1, 3.1]
        @test dp[dp.role .== "outcome", :lower] ≈ [1.0, 2.0, 3.0]
        @test all(ismissing, dp[dp.role .== "instrument", :lower])
        cm = MEM.CounterfactualMoments{Float64}([:y, :r],
            [1.0 0.2; 0.2 1.0], [0.8 0.1; 0.1 0.9], [1.0, 1.0], [0.9, 0.95],
            [1.0 0.2; 0.2 1.0], [1.0 0.1; 0.1 1.0], nothing, zeros(4, 2, 2),
            "rule", 4, 0.001, nothing)
        dm = DataFrame(cm)
        @test names(dm) == ["variable_i", "variable_j", "cov_base", "cov_cf", "corr_base", "corr_cf"]
        @test nrow(dm) == 4
        @test dm.cov_cf[2] ≈ 0.1
        @test dm[(dm.variable_i .== "y") .& (dm.variable_j .== "y"), :corr_base] == [1.0]
        ch = MEM.CounterfactualHistory{Float64}(["t1", "t2"], [:y, :r],
            [1.0 2.0; 3.0 4.0], [1.1 2.1; 3.1 4.1],
            cat([1.1 2.1; 3.1 4.1] .- 0.1, [1.1 2.1; 3.1 4.1] .+ 0.1; dims=3),
            reshape([0.1, 0.2], 1, 2), [0.01, 0.02], "rule", 4,
            [0.16, 0.84], 50, 1)
        dh = DataFrame(ch)
        @test names(dh) == ["date", "variable", "realized", "counterfactual", "cf_lower", "cf_upper", "rel_residual"]
        @test nrow(dh) == 4
        @test dh[dh.date .== "t2", :rel_residual] == [0.02, 0.02]
        @test dh.cf_upper ≈ dh.counterfactual .+ 0.1
        bp = MEM.BaselinePath{Float64}([:y], [:r], [[1.0, 2.0]], [[0.1, 0.2]],
            nothing, nothing, 2, "base")
        @test names(DataFrame(bp)) == ["period", "variable", "role", "value"]
        @test nrow(DataFrame(bp)) == 4
        pf = MEM.PolicyForecast{Float64}([:y, :r], [[1.0, 2.0], [3.0, 4.0]], nothing, 2, "2021Q2")
        dpf = DataFrame(pf)
        @test nrow(dpf) == 4
        @test dpf[dpf.variable .== "r", :value] ≈ [3.0, 4.0]
        sq = MEM.OPPSequence{Float64}(["t1", "t2"], reshape([1.0, 2.0], 1, 2),
            reshape([1.0, 2.0], 1, 2), reshape([0.1, 0.2], 1, 2),
            reshape([0.0, 0.0], 1, 2), reshape([0.0, 0.1], 1, 2),
            nothing, nothing, ["mp"], "quad")
        dsq = DataFrame(sq)
        @test names(dsq) == ["date", "shock", "delta", "delta_tc", "news", "pref", "aging"]
        @test dsq.delta ≈ [1.0, 2.0]
        fs = MEM.ForecastSufficiency{Float64}([:y, :r], [1.1 1.2; 1.0 1.1; 1.0 1.0],
            [1.05, 1.02], true, 3)
        dfs = DataFrame(fs)
        @test nrow(dfs) == 6
        @test dfs[dfs.observable .== "r", :one_step_ratio] ≈ fill(1.02, 3)
    end

    @testset "DataFrame(input-output results) (#859)" begin
        io = load_example(:wiot)
        lm = leontief(io)
        dl = DataFrame(lm)
        @test names(dl) == ["sector_i", "sector_j", "A", "L"]
        n = length(lm.x)
        @test nrow(dl) == n * n
        @test dl[(dl.sector_i .== io.sectors[1]) .& (dl.sector_j .== io.sectors[2]), :L] ≈ [lm.L[1, 2]]
        gm = ghosh(io)
        dg = DataFrame(gm)
        @test names(dg) == ["sector_i", "sector_j", "B", "G"]
        @test dg.G ≈ vec([gm.G[i, j] for i in 1:n for j in 1:n])
        lr = linkages(io)
        dlink = DataFrame(lr)
        @test names(dlink) == ["sector", "backward", "forward", "Ui", "Uj", "classification"]
        @test dlink.sector == io.sectors
        @test dlink.backward ≈ lr.backward
        mu = multipliers(io)
        dmu = DataFrame(mu)
        @test dmu.value ≈ mu.values
        @test dmu.kind == fill("output", n) && dmu.type == fill("I", n)
        fp = footprint(io, "CO2")
        dfp = DataFrame(fp)
        @test names(dfp) == ["stressor", "sector", "value", "total"]
        @test dfp.total ≈ repeat([sum(fp.total[i, :]) for i in 1:size(fp.total, 1)],
            inner=size(fp.by_sector, 2))
        io1 = IOData(io.Z .* 1.1, io.Y .* 1.1, io.va .* 1.1; sectors=io.sectors,
            regions=io.regions, fd_cats=io.fd_cats, va_cats=io.va_cats)
        sd = sda(io, io1)
        dsd = DataFrame(sd)
        @test "factor" in names(dsd) && "effect" in names(dsd)
        @test Set(dsd.factor) == Set(string.(keys(sd.effects)))
        @test dsd[dsd.factor .== string(first(sd.factors)), :effect] ≈ sd.effects[first(sd.factors)]
        # Regional footprint on a two-region toy.
        Z2 = [100.0 50.0; 0.0 50.0]
        Y2 = [30.0 20.0; 70.0 80.0]
        va2 = reshape([100.0, 100.0], 1, 2)
        io2 = IOData(Z2, Y2, va2; sectors=["USA_e", "CHN_e"], regions=["USA", "CHN"],
            fd_cats=["USA_fd", "CHN_fd"], va_cats=["VA"])
        add_extension!(io2, "co2", [5.0 8.0]; stressors=["CO2"], unit="Mt")
        rf = footprint(io2, "co2"; by=:region)
        drf = DataFrame(rf)
        @test names(drf) == ["stressor", "region", "production", "consumption"]
        @test nrow(drf) == 2
        @test drf.production ≈ vec(rf.production)
    end

    # ── Test battery, single-hypothesis rows (#860) ─────────────────────────────
    @testset "DataFrame(unit-root tests) (#860)" begin
        MEM = MacroEconometricModels
        rng = Xoshiro(26)
        y = randn(rng, 200)
        for (res, label) in ((adf_test(y), "ADF"), (kpss_test(y), "KPSS"), (pp_test(y), "Phillips-Perron"))
            df = DataFrame(res)
            @test names(df) == ["test", "statistic", "p_value", "decision", "cv_1pct", "cv_5pct", "cv_10pct"]
            @test df.test == [label]
            @test all(isfinite, [df.cv_1pct[1], df.cv_5pct[1], df.cv_10pct[1]])
            @test df.decision[1] in ("reject", "fail to reject")
        end
        za = MEM.ZAResult{Float64}(-4.5, 0.04, 50, 0.5, :both,
            Dict(1 => -5.0, 5 => -4.4, 10 => -4.1), 2, 100)
        dz = DataFrame(za)
        @test dz.decision == ["reject"]                     # -4.5 < -4.4 (left)
        @test (dz.break_index, dz.break_fraction) == ([50], [0.5])
        aw = MEM.AndrewsResult{Float64}(15.0, 0.01, 60, 0.6, :supwald,
            Dict(1 => 20.0, 5 => 15.5, 10 => 13.0), [1.0, 2.0], 0.15, 100, 3)
        da = DataFrame(aw)
        @test da.test == ["Andrews supwald"]
        @test da.decision == ["reject"]                     # p-based, like show()
        @test all(isfinite, [da.cv_1pct[1], da.cv_5pct[1], da.cv_10pct[1]])
        b2 = MEM.ADF2BreakResult{Float64}(-5.0, 0.02, 30, 70, 0.3, 0.7, 2, :level,
            Dict(1 => -5.5, 5 => -4.8, 10 => -4.5), 100)
        db2 = DataFrame(b2)
        @test db2.decision == ["reject"]
        @test (db2.break_1, db2.break_2) == ([30], [70])
        lm = MEM.LMUnitRootResult{Float64}(-3.0, 0.08, 1, [40], [0.4], 2, :constant,
            Dict(1 => -3.5, 5 => -2.8, 10 => -2.5), 100)
        dlm = DataFrame(lm)
        @test dlm.decision == ["reject"]                    # -3.0 < -2.8
        @test ismissing(dlm.break_2[1])                     # one break: padded
        fb = MEM.FactorBreakResult{Float64}(-2.0, 0.03, nothing, :han_inoue, 2, 100, 10, nothing, nothing)
        dfb = DataFrame(fb)
        @test dfb.test == ["Factor break han_inoue"]
        @test ismissing(dfb.break_index[1])
        ers = MEM.ERSResult{Float64}(2.5, 0.03, :constant, Dict(1 => 1.9, 5 => 2.9, 10 => 3.9), 100)
        @test DataFrame(ers).decision == ["reject"]         # 2.5 < 2.9 (left)
    end

    @testset "DataFrame(panel unit-root tests) (#860)" begin
        MEM = MacroEconometricModels
        llc = MEM.LLCResult{Float64}(-2.0, 0.023, -1.5, -0.05, 1.1, 0.5, 1.0, 95.5, [1, 1], :constant, 100, 2)
        dll = DataFrame(llc)
        @test dll.test == ["Levin-Lin-Chu"]
        @test dll.cv_5pct ≈ [-1.645]                        # N(0,1) CVs, like show()
        @test dll.decision == ["reject"]
        ips = MEM.IPSResult{Float64}(-1.0, 0.16, -1.8, [-1.7, -1.9], 0.0, 1.0, [1, 1], :constant, 100, 2)
        @test DataFrame(ips).decision == ["fail to reject"]
        br = MEM.BreitungPanelResult{Float64}(-2.5, 0.006, 1, :constant, 100, 2)
        @test DataFrame(br).decision == ["reject"]
        ha = MEM.HadriResult{Float64}(2.0, 0.023, 5.0, 1.0, 2.0, false, :constant, 100, 2)
        dh = DataFrame(ha)
        @test dh.cv_5pct ≈ [1.645]                          # right-tailed
        @test dh.decision == ["reject"]
        cips = MEM.PesaranCIPSResult{Float64}(-2.5, 0.01, [-2.4, -2.6],
            Dict(1 => -2.6, 5 => -2.2, 10 => -2.0), 1, :constant, 100, 2)
        dc = DataFrame(cips)
        @test dc.test == ["Pesaran CIPS"]
        @test dc.statistic == [-2.5]
        @test dc.decision == ["reject"]
    end

    @testset "DataFrame(serial, causality, model comparison) (#860)" begin
        MEM = MacroEconometricModels
        rng = Xoshiro(27)
        y = randn(rng, 200)
        @test DataFrame(ljung_box_test(y)).test == ["Ljung-Box"]
        @test DataFrame(box_pierce_test(y)).test == ["Box-Pierce"]
        @test DataFrame(durbin_watson_test(y)).test == ["Durbin-Watson"]
        @test DataFrame(fisher_test(y)).test == ["Fisher periodicity"]
        @test isfinite(DataFrame(fisher_test(y)).peak_freq[1])
        @test DataFrame(bartlett_white_noise_test(y)).test == ["Bartlett white noise"]
        vm = estimate_var(randn(rng, 60, 2), 1)
        g = granger_test(vm, 1, 2)
        dg = DataFrame(g)
        @test dg.cause == ["1"] && dg.effect == [2]
        lrt = MEM.LRTestResult{Float64}(5.0, 0.08, 2, -100.0, -97.5, 3, 5, 100, 100)
        @test DataFrame(lrt).decision == ["fail to reject"]
        lmt = MEM.LMTestResult{Float64}(7.0, 0.03, 2, 100, 2.6)
        @test DataFrame(lmt).decision == ["reject"]
    end

    @testset "DataFrame(cointegration stability tests) (#860)" begin
        MEM = MacroEconometricModels
        eg = MEM.EngleGrangerResult{Float64}(-3.5, 0.02, 2, :constant, 1, 2, 100)
        @test DataFrame(eg).decision == ["reject"]
        hi = MEM.HansenInstabilityResult{Float64}(0.8, 0.01, :constant, :none, 3, 1, 100)
        @test DataFrame(hi).test == ["Hansen instability"]
        pa = MEM.ParkAddedResult{Float64}(9.0, 0.03, 2, 1, :constant, :constant, 1, 100)
        @test DataFrame(pa).decision == ["reject"]
    end

    @testset "DataFrame(test-battery singles) (#860)" begin
        MEM = MacroEconometricModels
        rng = Xoshiro(28)
        bub = MEM.BubbleResult{Float64}(:gsadf, 2.5, 0.01, Dict(1 => 2.0, 5 => 1.5, 10 => 1.2),
            [1.0, 2.0], [1.0, 1.5], [1, 2], [(5, 9)], 0.1, 1, :mc, 2000, 100)
        db = DataFrame(bub)
        @test db.test == ["GSADF"]
        @test db.decision == ["reject"]                     # 2.5 > 1.5 (right)
        ed = MEM.EDFTestResult{Float64}(:ad, :normal, :estimate, 1.2, 1.1, 0.04, 100,
            [0.0, 1.0], Dict(1 => 1.5, 5 => 1.0, 10 => 0.8), "case A")
        de = DataFrame(ed)
        @test de.decision == ["reject"]                     # p-based
        @test de.raw_statistic == [1.1]
        ed2 = MEM.EDFTestResult{Float64}(:ks, :normal, :specified, 0.5, 0.5, NaN, 100,
            Float64[], Dict{Int,Float64}(), "case B")
        de2 = DataFrame(ed2)
        @test ismissing(de2.p_value[1]) && ismissing(de2.decision[1]) && ismissing(de2.cv_5pct[1])
        y = randn(rng, 100)
        g = vcat(fill(1, 50), fill(2, 50))
        @test DataFrame(equality_test(y, g; test=:t)).test == ["Two-Sample t-Test (pooled)"]
        @test DataFrame(cor_test(y, randn(rng, 100))).test == ["Correlation (Pearson)"]
        @test isfinite(DataFrame(cor_test(y, randn(rng, 100))).ci_lower[1])
        X = hcat(ones(100), randn(rng, 100, 2))
        w = white_test(randn(rng, 100), X)
        dw = DataFrame(w)
        @test dw.aux_r2[1] >= 0.0
        dm = diebold_mariano(randn(rng, 100), randn(rng, 100))
        dd = DataFrame(dm)
        @test dd.test == ["Diebold-Mariano"]
        @test isfinite(dd.dbar[1]) && isfinite(dd.lrvar[1])
        pt = MEM.PanelTestResult{Float64}("Hausman test", 12.0, 0.01, 3, "reject RE")
        @test DataFrame(pt).decision == ["reject"]
        pv = MEM.PVARTestResult{Float64}("Hansen J-test", 5.0, 0.2, 4, 10, 6)
        @test DataFrame(pv).decision == ["fail to reject"]
        nt = MEM.NormalityTestResult{Float64}(:jarque_bera, 8.0, 0.02, 2, 3, 200, nothing, nothing)
        @test endswith(DataFrame(nt).test[1], "(multivariate)")
        cw = MEM.ClarkWestResult{Float64}(1.8, 0.036, 0.05, 0.001, 1, :greater, 200)
        dcw = DataFrame(cw)
        @test dcw.decision == ["reject"] && dcw.fbar == [0.05]
        fe = MEM.ForecastEncompassingResult{Float64}(0.7, 0.3, 0.1, 3.0, 0.003, 2, :bartlett, 200)
        dfe = DataFrame(fe)
        @test dfe.statistic == [3.0] && dfe.b1 == [0.7]
    end

    # ── write_csv ───────────────────────────────────────────────────────────────
    @testset "write_csv round-trips through a co-author read-back" begin
        rng = Xoshiro(19)
        X = randn(rng, 80, 2); y = X * [1.0, -0.5] .+ randn(rng, 80)
        m = estimate_reg(y, X)
        path = tempname() * ".csv"
        @test write_csv(m, path) == path
        raw, hdr = readdlm(path, ',', header=true)
        @test vec(hdr) == ["term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        @test size(raw, 1) == length(coef(m))
        @test Float64.(raw[:, 2]) ≈ coef(m)

        # Passing an array-valued result directly routes through long_table.
        vm = estimate_var(randn(rng, 100, 2), 2)
        ipath = tempname() * ".csv"
        write_csv(irf(vm, 5; method=:cholesky), ipath)
        iraw, ihdr = readdlm(ipath, ',', header=true)
        @test vec(ihdr) == ["horizon", "variable", "shock", "value", "lower", "upper"]
        @test size(iraw, 1) == 5 * 2 * 2
    end
end
