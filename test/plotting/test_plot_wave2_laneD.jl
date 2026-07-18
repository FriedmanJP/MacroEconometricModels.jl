# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Wave-2 Lane D plotting tests: PLT-32 (non-Gaussian / heteroskedastic SVAR),
# PLT-33 (DSGE extras — PF path / smoother / GMM estimation / Bayesian sim),
# PLT-36 (micro / panel / LDV coefficient forest plots). Uses the shared
# assertions in plot_test_helpers.jl (Testing Rules 1-7): check_plot,
# assert_all_json_valid, assert_escapes, assert_nan_becomes_null,
# series_count/series_names, panel_titles, HOSTILE_NAME.
# =============================================================================

using Test, Random, DataFrames, LinearAlgebra, Distributions

# Self-bootstrap the shared helpers when run standalone.
if !isdefined(@__MODULE__, :check_plot)
    include(joinpath(@__DIR__, "plot_test_helpers.jl"))
end

const _MEM = MacroEconometricModels

# -----------------------------------------------------------------------------
# Fixture builders
# -----------------------------------------------------------------------------

# A 2-variable statistical-identification stack (directly constructed so NaN edge
# cases are controllable). `bad` injects a NaN into a data field per type.
function _laneD_ms(; K=2, Tt=5, nanprob=false, nantrans=false)
    B0 = [1.0 0.2; 0.3 1.0]; Q = Matrix(1.0I, 2, 2)
    rp = zeros(Tt, K)
    for t in 1:Tt
        v = abs.(randn(MersenneTwister(t), K)) .+ 0.1
        rp[t, :] = v ./ sum(v)
    end
    nanprob && (rp[1, 1] = NaN)
    tm = K == 2 ? [0.8 0.2; 0.3 0.7] : (x = rand(K, K); x ./ sum(x; dims=2))
    nantrans && (tm[1, 1] = NaN)
    _MEM.MarkovSwitchingSVARResult(B0, Q, [Matrix(1.0I, 2, 2), Matrix(2.0I, 2, 2)],
        [[1.0, 1.0], [1.5, 2.0]], rp, tm, -100.0, true, 5, K)
end

function _laneD_garch(; nanvar=false)
    B0 = [1.0 0.2; 0.3 1.0]; Q = Matrix(1.0I, 2, 2)
    cv = abs.(randn(MersenneTwister(3), 6, 2)) .+ 0.2
    nanvar && (cv[2, 1] = NaN)
    _MEM.GARCHSVARResult(B0, Q, [0.1 0.2 0.7; 0.1 0.3 0.6], cv,
        randn(MersenneTwister(4), 6, 2), -50.0, true, 3)
end

function _laneD_star()
    B0 = [1.0 0.2; 0.3 1.0]; Q = Matrix(1.0I, 2, 2)
    sv = collect(-2.0:0.5:2.0)
    gv = 1.0 ./ (1 .+ exp.(-2.5 .* (sv .- 0.1)))
    _MEM.SmoothTransitionSVARResult(B0, Q, [Matrix(1.0I, 2, 2)], [[1.0, 1.0]],
        2.5, 0.1, sv, gv, -60.0, true, 4)
end

_laneD_extvol() = _MEM.ExternalVolatilitySVARResult([1.0 0.2; 0.3 1.0], Matrix(1.0I, 2, 2),
    [Matrix(1.0I, 2, 2), Matrix(2.0I, 2, 2)], [[1.0, 1.0], [1.5, 2.0]],
    [[1, 2, 3], [4, 5, 6]], -70.0)

_laneD_ica() = _MEM.ICASVARResult([1.0 0.2; 0.3 1.0], [1.0 -0.2; -0.3 1.0],
    Matrix(1.0I, 2, 2), randn(MersenneTwister(5), 6, 2), :fastica, true, 10, 0.5)

_laneD_ml() = _MEM.NonGaussianMLResult([1.0 0.2; 0.3 1.0], Matrix(1.0I, 2, 2),
    randn(MersenneTwister(6), 6, 2), :t, -40.0, -55.0, Dict{Symbol,Any}(),
    Matrix(1.0I, 4, 4), [0.1 0.1; 0.1 0.1], true, 8, 90.0, 100.0)

# Directly-constructed DSGE fixtures.
function _laneD_spec(varnames::Vector{Symbol}, ss::Vector{Float64})
    n = length(varnames)
    eqs = Expr[:($(v)[t] = $(v)[t]) for v in varnames]
    _MEM.DSGESpec{Float64}(varnames, [:e], Symbol[], Dict{Symbol,Float64}(),
        eqs, Function[identity for _ in 1:n], 0, Int[], ss)
end

function _laneD_pf(; nvar=2, nanpath=false)
    varnames = [Symbol("v$i") for i in 1:nvar]
    ss = Float64.(1:nvar)
    spec = _laneD_spec(varnames, ss)
    path = zeros(4, nvar)
    for j in 1:nvar, t in 1:4
        path[t, j] = ss[j] + 0.1 * sin(t + j)
    end
    nanpath && (path[2, 1] = NaN)
    _MEM.PerfectForesightPath(path, path .- ss', true, 7, spec)
end

function _laneD_ks(; ns=2, Tn=6, nancov=false)
    states = randn(MersenneTwister(7), ns, Tn)
    covs = zeros(ns, ns, Tn)
    for t in 1:Tn
        covs[:, :, t] = Matrix(0.5I, ns, ns)
    end
    nancov && (states[1, 2] = NaN)
    _MEM.KalmanSmootherResult(states, covs, randn(MersenneTwister(8), ns, Tn),
        states, covs, states, covs, -123.0)
end

function _laneD_bsim(; nv=2, nanq=false)
    Tp, nq = 8, 4
    qs = zeros(Tp, nv, nq)
    for t in 1:Tp, v in 1:nv
        qs[t, v, :] = sort(randn(MersenneTwister(t + v), nq)) .+ v
    end
    nanq && (qs[1, 1, 1] = NaN)
    pe = randn(MersenneTwister(9), Tp, nv)
    _MEM.BayesianDSGESimulation(qs, pe, Tp, ["y$i" for i in 1:nv],
        [0.05, 0.16, 0.84, 0.95], randn(MersenneTwister(10), 20, Tp, nv))
end

# DSGEEstimation needs a real solution object for its Union-typed field.
function _laneD_est(; hostile=false)
    spec = @dsge begin
        parameters: rho = 0.5
        endogenous: y
        exogenous: eps
        y[t] = rho * y[t-1] + eps[t]
    end
    sol = solve(spec)
    names = hostile ? [Symbol(HOSTILE_NAME), :beta] : [:rho, :beta]
    _MEM.DSGEEstimation{Float64}([0.5, 0.9], [0.01 0.0; 0.0 0.02], names,
        :euler_gmm, 3.2, 0.35, sol, true, spec)
end

# Small panels / cross-sections for the micro coefficient plots.
function _laneD_panel(; N=8, Tt=10, varnames=("x1", "x2"), seed=11)
    rng = MersenneTwister(seed)
    df = DataFrame(id=repeat(1:N, inner=Tt), time=repeat(1:Tt, outer=N))
    df[!, varnames[1]] = randn(rng, N * Tt)
    df[!, varnames[2]] = randn(rng, N * Tt)
    df.y = 0.5 .* df[!, varnames[1]] .- 0.3 .* df[!, varnames[2]] .+ randn(rng, N * Tt)
    xtset(df, :id, :time)
end

# =============================================================================
@testset "Wave-2 Lane D — statid SVAR / DSGE extras / micro coef" begin

    # =========================================================================
    # PLT-32 — non-Gaussian / heteroskedastic SVAR
    # =========================================================================
    @testset "PLT-32 statistical-identification SVAR" begin

        @testset "MarkovSwitching views + content" begin
            ms = _laneD_ms()
            for v in (:regimes, :B0, :transition)
                p = plot_result(ms; view=v)
                check_plot(p); assert_all_json_valid(p)
            end
            pr = plot_result(ms; view=:regimes)
            @test occursin("Regime Probabilities", pr.html)
            @test series_names(pr.html) == ["Regime 1", "Regime 2"]   # stacked-area layers
            @test_throws ArgumentError plot_result(ms; view=:bogus)
        end

        @testset "MarkovSwitching heatmap scales (signed vs nonneg)" begin
            ms = _laneD_ms()
            pB = plot_result(ms; view=:B0)
            @test occursin("interpolateRdBu", pB.html)          # B0 signed → diverging
            pT = plot_result(ms; view=:transition)
            @test occursin("interpolateBlues", pT.html)         # transition nonneg → sequential
            @test occursin("missing", pT.html)                  # heatmap color-scale legend
        end

        @testset "MarkovSwitching NaN → null (prob + transition)" begin
            assert_nan_becomes_null(plot_result(_laneD_ms(nanprob=true); view=:regimes))
            assert_nan_becomes_null(plot_result(_laneD_ms(nantrans=true); view=:transition))
        end

        @testset "GARCH-SVAR variance / shocks" begin
            g = _laneD_garch()
            pv = plot_result(g; view=:variance)
            check_plot(pv); assert_all_json_valid(pv)
            @test series_names(pv.html) == ["Shock 1", "Shock 2"]
            @test occursin("Conditional Variances", pv.html)
            ps = plot_result(g; view=:shocks)
            check_plot(ps)
            @test_throws ArgumentError plot_result(g; view=:bogus)
            assert_nan_becomes_null(plot_result(_laneD_garch(nanvar=true); view=:variance))
        end

        @testset "SmoothTransition function + threshold vline + γ title" begin
            p = plot_result(_laneD_star())
            check_plot(p); assert_all_json_valid(p)
            @test occursin("Transition Function", p.html)
            @test occursin("γ = 2.5", p.html)                   # γ rounded via _fmt (C9)
            @test occursin("\"axis\":\"x\"", p.html)            # threshold vertical ref line
        end

        @testset "ExternalVolatility regime membership" begin
            p = plot_result(_laneD_extvol())
            check_plot(p); assert_all_json_valid(p)
            @test occursin("Regime Membership", p.html)
        end

        @testset "ICA views (mixing / unmixing / shocks) + method in title" begin
            ica = _laneD_ica()
            for v in (:mixing, :unmixing, :shocks)
                p = plot_result(ica; view=v)
                check_plot(p); assert_all_json_valid(p)
            end
            @test occursin("fastica", plot_result(ica; view=:mixing).html)
            @test occursin("interpolateRdBu", plot_result(ica; view=:mixing).html)  # B0 signed
            @test_throws ArgumentError plot_result(ica; view=:bogus)
        end

        @testset "NonGaussianML B0 + LR annotation" begin
            p = plot_result(_laneD_ml())
            check_plot(p); assert_all_json_valid(p)
            @test occursin("LR vs Gaussian = 30", p.html)       # 2*(-40 - -55) = 30, _fmt (C9)
            @test occursin("interpolateRdBu", p.html)
        end
    end

    # =========================================================================
    # PLT-33 — DSGE additions
    # =========================================================================
    @testset "PLT-33 DSGE extras" begin

        @testset "PerfectForesightPath levels/deviations + SS ref + var select" begin
            pf = _laneD_pf(nvar=3)
            pl = plot_result(pf; view=:levels)
            check_plot(pl); assert_all_json_valid(pl)
            @test length(panel_titles(pl.html)) == 3            # one panel per variable
            @test occursin("Transition Path (levels)", pl.html)
            @test occursin("\"axis\":\"y\"", pl.html)           # steady-state ref line
            pd = plot_result(pf; view=:deviations)
            check_plot(pd)
            @test occursin("deviations from SS", pd.html)
            # var selection Int + String
            @test length(panel_titles(plot_result(pf; var=1).html)) == 1
            @test length(panel_titles(plot_result(pf; var="v2").html)) == 1
            @test_throws ArgumentError plot_result(pf; var="nope")
            @test_throws ArgumentError plot_result(pf; var=99)
            @test_throws ArgumentError plot_result(pf; view=:bad)
            assert_nan_becomes_null(plot_result(_laneD_pf(nanpath=true); view=:levels))
        end

        @testset "PerfectForesightPath panel cap (C7)" begin
            pf = _laneD_pf(nvar=4)
            p = plot_result(pf; max_panels=2)
            @test length(panel_titles(p.html)) == 2
            @test occursin("Showing 2 of 4 variables", p.html)  # cap note visible
        end

        @testset "KalmanSmoother states + band + required varnames" begin
            ks = _laneD_ks(ns=2, Tn=6)
            p = plot_result(ks; varnames=["a", "b"])
            check_plot(p); assert_all_json_valid(p)
            @test length(panel_titles(p.html)) == 2
            @test occursin("Smoothed States", p.html)
            @test occursin("\"lo_key\":\"lo\"", p.html)          # ±1.96 s.e. band
            # observed overlay
            po = plot_result(ks; varnames=["a", "b"], data=randn(MersenneTwister(1), 6, 2))
            @test occursin("Observed", po.html)
            # required + validated kwargs
            @test_throws ArgumentError plot_result(ks)                       # missing varnames
            @test_throws ArgumentError plot_result(ks; varnames=["a"])       # wrong length
            @test_throws ArgumentError plot_result(ks; varnames=["a", "b"], data=randn(3, 3))
            # var selection
            @test length(panel_titles(plot_result(ks; varnames=["a", "b"], var="b").html)) == 1
            assert_nan_becomes_null(plot_result(_laneD_ks(nancov=true); varnames=["a", "b"]))
        end

        @testset "DSGEEstimation parameter coef plot + J-test note" begin
            est = _laneD_est()
            p = plot_result(est)
            check_plot(p); assert_all_json_valid(p)
            @test occursin("Estimated Parameters", p.html)
            @test occursin("J = 3.2", p.html)                    # J-stat rounded (C9)
            @test occursin("p = 0.35", p.html)                   # J p-value
            # hostile param name survives every sink
            assert_escapes(plot_result(_laneD_est(hostile=true)))
        end

        @testset "BayesianDSGESimulation fan + var select + caps" begin
            sim = _laneD_bsim(nv=2)
            p = plot_result(sim)
            check_plot(p); assert_all_json_valid(p)
            @test length(panel_titles(p.html)) == 2
            @test occursin("Posterior-Predictive", p.html)
            @test occursin("\"lo_key\"", p.html)                 # nested quantile bands
            @test length(panel_titles(plot_result(sim; var=1).html)) == 1
            @test length(panel_titles(plot_result(sim; var="y2").html)) == 1
            @test_throws ArgumentError plot_result(sim; var="nope")
            # hostile variable name
            simh = _laneD_bsim(nv=1)
            simh2 = _MEM.BayesianDSGESimulation(simh.quantiles, simh.point_estimate,
                simh.T_periods, [HOSTILE_NAME], simh.quantile_levels, simh.all_paths)
            assert_escapes(plot_result(simh2))
            assert_nan_becomes_null(plot_result(_laneD_bsim(nanq=true)))
            # cap note
            pc = plot_result(_laneD_bsim(nv=4); max_panels=2)
            @test occursin("Showing 2 of 4 variables", pc.html)
        end
    end

    # =========================================================================
    # PLT-36 — micro / panel / LDV coefficient plots
    # =========================================================================
    @testset "PLT-36 micro / panel coefficient plots" begin
        rng = MersenneTwister(21)
        n = 300
        X = randn(rng, n, 2)
        xb = X * [0.8, -0.5]

        @testset "PanelReg / PanelIV coef plots" begin
            pd = _laneD_panel()
            preg = estimate_xtreg(pd, :y, [:x1, :x2])
            p = plot_result(preg)
            check_plot(p); assert_all_json_valid(p)
            @test occursin("Panel Regression Coefficients", p.html)
            @test occursin("ci_lo", p.html)                      # dot-and-whisker rows
            # hostile varname round-trips through the coef data literal
            pdh = _laneD_panel(varnames=(HOSTILE_NAME, "x2"))
            assert_escapes(plot_result(estimate_xtreg(pdh, :y, [Symbol(HOSTILE_NAME), :x2])))
            # PanelIV — x2 endogenous, z the instrument
            rng2 = MersenneTwister(31)
            Np, Tt = 8, 10
            dfi = DataFrame(id=repeat(1:Np, inner=Tt), time=repeat(1:Tt, outer=Np))
            dfi.x1 = randn(rng2, Np * Tt); dfi.z = randn(rng2, Np * Tt)
            dfi.x2 = 0.7 .* dfi.z .+ randn(rng2, Np * Tt)
            dfi.y = 0.5 .* dfi.x1 .+ 0.4 .* dfi.x2 .+ randn(rng2, Np * Tt)
            piv = estimate_xtiv(xtset(dfi, :id, :time), :y, [:x1], [:x2]; instruments=[:z])
            check_plot(plot_result(piv))
        end

        @testset "Ordered logit coef + cutpoint note" begin
            yo = [xb[i] + randn(rng) < -0.5 ? 1 : xb[i] + randn(rng) < 0.5 ? 2 : 3 for i in 1:n]
            om = estimate_ologit(yo, X; varnames=["x1", "x2"])
            p = plot_result(om)
            check_plot(p); assert_all_json_valid(p)
            @test occursin("Ordered Model Coefficients", p.html)
            @test any(t -> occursin("Cutpoints", t), panel_titles(p.html))  # cutpoint panel
        end

        @testset "Multinomial logit per-outcome facets + category select" begin
            ym = rand(rng, 1:3, n)
            mm = estimate_mlogit(ym, X; varnames=["x1", "x2"])
            p = plot_result(mm)
            check_plot(p); assert_all_json_valid(p)
            @test length(panel_titles(p.html)) == 2              # J-1 = 2 non-base outcomes
            @test any(t -> occursin("(vs ", t), panel_titles(p.html))
            # category selection Int + label
            @test length(panel_titles(plot_result(mm; category=1).html)) == 1
            @test length(panel_titles(plot_result(mm; category="2").html)) == 1
            @test_throws ArgumentError plot_result(mm; category="nope")
        end

        @testset "Multinomial marginal effects — dots only when SE missing (C6)" begin
            ym = rand(rng, 1:3, n)
            me = marginal_effects(estimate_mlogit(ym, X; varnames=["x1", "x2"]))
            @test me.se === nothing
            p = plot_result(me)
            check_plot(p); assert_all_json_valid(p)
            @test occursin("points only", p.html)                # honest subtitle (C6)
        end

        @testset "OddsRatio forest plot (log x, ref = 1)" begin
            yb = Int.(xb .+ randn(rng, n) .> 0)
            orr = odds_ratio(estimate_logit(yb, X))
            p = plot_result(orr)
            check_plot(p); assert_all_json_valid(p)
            @test occursin("scaleLog", p.html)                   # log x-axis (A4 logx option)
            @test occursin("ref = 1", p.html)
            # NaN CI → null (directly-constructed OddsRatio)
            orn = _MEM.OddsRatio([1.2, 0.8], [0.1, 0.1], [1.0, NaN], [1.5, 1.0],
                ["x1", "x2"], 0.95)
            assert_nan_becomes_null(plot_result(orn))
        end

        @testset "Heckman two-equation view dispatch" begin
            Z = hcat(X, randn(rng, n))
            d = Int.(Z * [0.5, -0.3, 0.4] .+ randn(rng, n) .> 0)
            hk = estimate_heckman(xb .+ randn(rng, n), X, d, Z)
            check_plot(plot_result(hk; view=:outcome))
            check_plot(plot_result(hk; view=:selection))
            @test occursin("Selection Equation", plot_result(hk; view=:selection).html)
            @test_throws ArgumentError plot_result(hk; view=:bogus)
        end

        @testset "Tobit / TruncReg / CointReg coef plots" begin
            tb = estimate_tobit(max.(xb .+ randn(rng, n), 0.0), X; lower=0.0)
            check_plot(plot_result(tb)); assert_all_json_valid(plot_result(tb))
            tr = estimate_truncreg(xb .+ randn(rng, n) .+ 5.0, X; lower=0.0)
            check_plot(plot_result(tr))
            Tc = 100
            xc = cumsum(randn(rng, Tc))
            cr = estimate_cointreg(1.5 .* xc .+ randn(rng, Tc), xc)
            check_plot(plot_result(cr))
        end

        @testset "PMG long-run coefficients" begin
            Np, Tp2 = 6, 30
            dfp = DataFrame(id=repeat(1:Np, inner=Tp2), time=repeat(1:Tp2, outer=Np))
            dfp.x = randn(rng, Np * Tp2)
            dfp.y = zeros(Np * Tp2)
            for i in 1:Np
                off = (i - 1) * Tp2
                for t in 2:Tp2
                    dfp.y[off+t] = 0.5 * dfp.y[off+t-1] + 0.4 * dfp.x[off+t] + 0.1 * randn(rng)
                end
            end
            pmg = estimate_pmg(xtset(dfp, :id, :time), :y, :x)
            p = plot_result(pmg)
            check_plot(p)
            @test occursin("Long-run Coefficients", p.html)
        end

        @testset "SUR / 3SLS one panel per equation" begin
            Ts = 80
            X1 = hcat(ones(Ts), randn(rng, Ts)); X2 = hcat(ones(Ts), randn(rng, Ts))
            y1 = X1 * [1.0, 0.5] .+ randn(rng, Ts); y2 = X2 * [0.5, -0.3] .+ randn(rng, Ts)
            sur = estimate_sur([(y1, X1), (y2, X2)])
            p = plot_result(sur)
            check_plot(p); assert_all_json_valid(p)
            @test length(panel_titles(p.html)) == 2              # one panel per equation
            tsls = estimate_3sls([(y1, X1), (y2, X2)], hcat(ones(Ts), randn(rng, Ts, 3)))
            @test length(panel_titles(plot_result(tsls).html)) == 2
        end
    end
end
