# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.
#
# EV-39 (#447): Forecast evaluation suite — accuracy metrics, Diebold–Mariano /
# Clark–West / Mincer–Zarnowitz / forecast-encompassing tests, and combination.
#
# Oracle strategy (no invented numerics):
#   (1) Cross-implementation known values. R's forecast::dm.test is not
#       installable in this environment (fortran toolchain missing), so the
#       reference statistics are produced by a FAITHFUL base-R transcription of
#       the published forecast::dm.test source (default varestimator="acf",
#       HLN 1997 small-sample factor, t_{T-1}) on a DETERMINISTIC dataset built
#       from sin/cos (IEEE-754-identical across R and Julia). The exact R code
#       is reproduced below. Matched to 1e-4 (actually ~1e-10). Likewise the
#       Mincer–Zarnowitz HAC Wald and the HLN encompassing t come from a base-R
#       Newey–West sandwich transcription (S built WITHOUT ÷n, bread (X'X)^{-1},
#       bartlett weights 1−j/(L+1)), matching src/core/covariance.jl newey_west.
#   (2) Analytic identities: Theil U2 = 1 exactly for the naive no-change
#       forecast; the Theil MSE bias/variance/covariance proportions sum to 1.
#   (3) Property: the equal-weights combination beats the WORST individual model
#       on RMSE (seeded 3-model DGP); Granger–Ramanathan in-sample RMSE ≤ equal.
#
#   R reference (Rscript, base R):
#     dm.test <- function(e1,e2,alternative="two.sided",h=1,power=2){
#       d <- c(abs(e1)^power - abs(e2)^power)
#       d.cov <- acf(d,na.action=na.omit,lag.max=h-1,type="covariance",plot=FALSE)$acf[,,1]
#       n <- length(d); d.var <- sum(c(d.cov[1],2*d.cov[-1]))/n
#       STAT <- mean(d)/sqrt(d.var)
#       k <- ((n+1-2*h+(h/n)*(h-1))/n)^(1/2); STAT <- STAT*k
#       PVAL <- 2*pt(-abs(STAT), df=n-1); list(stat=STAT,p=PVAL) }
#     T<-100; t<-1:T
#     e1<-sin(0.3*t)+0.5*cos(0.1*t); e2<-0.8*sin(0.3*t+0.5)+0.2
#     -> h=1,power=1: DM=3.7679806064 p=0.0002797289
#        h=1,power=2: DM=4.6332224783 p=0.0000109800
#        h=4,power=1: DM=1.7155284767 p=0.0893765404
#        h=4,power=2: DM=2.0136524733 p=0.0467599124
#     actual<-2+0.5*sin(0.2*t)+cos(0.15*t); fc<-actual+e1; L=4 bartlett NW:
#        MZ a=1.0153821443 b=0.5072146708 Wald=56.6178338080 se_a=0.2016127303 se_b=0.0668885259
#     f1<-actual+e1; f2<-actual+e2; enc b2=0.8139118702 se=0.1795917105 t=4.5320124621

using Test, MacroEconometricModels, Random, LinearAlgebra, Statistics

@testset "EV-39 Forecast Evaluation Suite" begin
    # Deterministic, cross-language-identical inputs
    t = collect(1.0:100.0)
    e1 = sin.(0.3 .* t) .+ 0.5 .* cos.(0.1 .* t)
    e2 = 0.8 .* sin.(0.3 .* t .+ 0.5) .+ 0.2
    actual = 2.0 .+ 0.5 .* sin.(0.2 .* t) .+ cos.(0.15 .* t)
    fc = actual .+ e1
    f1 = actual .+ e1
    f2 = actual .+ e2
    atol4 = 1e-4

    @testset "Diebold–Mariano vs R forecast::dm.test (HLN, t_{T-1})" begin
        r11 = diebold_mariano(e1, e2; h=1, loss=:ad)
        @test isapprox(r11.statistic, 3.7679806064; atol=atol4)
        @test isapprox(r11.pvalue,    0.0002797289; atol=atol4)

        r12 = diebold_mariano(e1, e2; h=1, loss=:se)
        @test isapprox(r12.statistic, 4.6332224783; atol=atol4)
        @test isapprox(r12.pvalue,    0.0000109800; atol=atol4)

        r41 = diebold_mariano(e1, e2; h=4, loss=:ad)
        @test isapprox(r41.statistic, 1.7155284767; atol=atol4)
        @test isapprox(r41.pvalue,    0.0893765404; atol=atol4)

        r42 = diebold_mariano(e1, e2; h=4, loss=:se)
        @test isapprox(r42.statistic, 2.0136524733; atol=atol4)
        @test isapprox(r42.pvalue,    0.0467599124; atol=atol4)

        # StatsAPI + display
        @test MacroEconometricModels.StatsAPI.pvalue(r12) == r12.pvalue
        @test occursin("Diebold", sprint(show, r12))
    end

    @testset "DM without HLN uses N(0,1)" begin
        rt = diebold_mariano(e1, e2; h=1, loss=:se, hln=true)
        rn = diebold_mariano(e1, e2; h=1, loss=:se, hln=false)
        # HLN scales the h=1 stat by sqrt((n-1)/n)
        n = length(e1)
        @test isapprox(rt.statistic, rn.statistic * sqrt((n - 1) / n); rtol=1e-10)
        # normal p-value is smaller (thinner tails than t_{99})
        @test rn.pvalue < rt.pvalue
        # custom loss function equals :se
        rc = diebold_mariano(e1, e2; h=1, loss=(x -> x^2))
        @test isapprox(rc.statistic, rt.statistic; rtol=1e-12)
        @test rc.loss == :custom
    end

    @testset "DM alternatives" begin
        rg = diebold_mariano(e1, e2; h=1, loss=:se, alternative=:greater)
        rl = diebold_mariano(e1, e2; h=1, loss=:se, alternative=:less)
        rtwo = diebold_mariano(e1, e2; h=1, loss=:se, alternative=:two_sided)
        @test isapprox(rg.pvalue + rl.pvalue, 1.0; atol=1e-10)
        @test isapprox(rtwo.pvalue, 2 * rg.pvalue; atol=1e-10)   # positive stat
    end

    @testset "Point metrics + identities" begin
        ev = forecast_evaluate(actual, hcat(f1, f2); model_names=["m1", "m2"])
        @test ev.metrics == ["ME", "MAE", "RMSE", "MAPE", "sMAPE", "MASE", "U1", "U2"]
        @test size(ev.values) == (2, 8)

        # RMSE reconstruction
        rmse1 = sqrt(mean((actual .- f1).^2))
        k_rmse = findfirst(==("RMSE"), ev.metrics)
        @test isapprox(ev.values[1, k_rmse], rmse1; rtol=1e-12)

        # Theil MSE decomposition proportions sum to 1
        @test isapprox(sum(ev.decomp[1, :]), 1.0; atol=1e-10)
        @test isapprox(sum(ev.decomp[2, :]), 1.0; atol=1e-10)

        # Theil U2 == 1 EXACTLY for the naive no-change forecast fc_t = actual_{t-1}
        naive = vcat(actual[1], actual[1:end-1])
        evn = forecast_evaluate(actual, naive)
        k_u2 = findfirst(==("U2"), ev.metrics)
        @test isapprox(evn.values[1, k_u2], 1.0; atol=1e-12)

        # U1 bounded in [0,1]; MASE of a perfect forecast is 0
        k_u1 = findfirst(==("U1"), ev.metrics)
        @test 0 <= ev.values[1, k_u1] <= 1
        evp = forecast_evaluate(actual, actual)
        k_mase = findfirst(==("MASE"), ev.metrics)
        @test isapprox(evp.values[1, k_mase], 0.0; atol=1e-12)

        # Display + refs render
        @test occursin("Forecast Evaluation", sprint(show, ev))
        @test occursin("Theil", sprint(show, ev))
        @test !isempty(refs(ev))
    end

    @testset "MAPE/sMAPE zero-actual guard" begin
        a = [0.0, 1.0, 2.0, 3.0, 4.0]
        f = [0.1, 1.1, 1.9, 3.2, 3.8]
        ev = forecast_evaluate(a, f)
        k_mape = findfirst(==("MAPE"), ev.metrics)
        @test isfinite(ev.values[1, k_mape])   # zero-actual row skipped, not Inf
    end

    @testset "Mincer–Zarnowitz HAC Wald vs base-R sandwich (L=4)" begin
        mz = mincer_zarnowitz(actual, fc; lags=4, kernel=:bartlett)
        @test isapprox(mz.a, 1.0153821443; atol=atol4)
        @test isapprox(mz.b, 0.5072146708; atol=atol4)
        @test isapprox(mz.se[1], 0.2016127303; atol=atol4)
        @test isapprox(mz.se[2], 0.0668885259; atol=atol4)
        @test isapprox(mz.wald, 56.6178338080; atol=1e-3)
        @test isapprox(mz.fstat, 56.6178338080 / 2; atol=1e-3)
        @test occursin("Mincer", sprint(show, mz))

        # An (almost) efficient forecast: a≈0, b≈1, Wald small
        mz_eff = mincer_zarnowitz(actual, actual .+ 1e-6 .* e1; lags=0)
        @test isapprox(mz_eff.a, 0.0; atol=1e-3)
        @test isapprox(mz_eff.b, 1.0; atol=1e-3)
    end

    @testset "Forecast encompassing (HLN 1998) vs base-R" begin
        enc = forecast_encompassing(actual, f1, f2; lags=4, kernel=:bartlett)
        @test isapprox(enc.b2, 0.8139118702; atol=atol4)
        @test isapprox(enc.se_b2, 0.1795917105; atol=atol4)
        @test isapprox(enc.tstat, 4.5320124621; atol=atol4)
        @test enc.pvalue < 0.01   # f1 does NOT encompass f2 here
        @test occursin("Encompassing", sprint(show, enc))
    end

    @testset "Clark–West nested-model test" begin
        # Nested DGP: small = AR(0) mean forecast; big adds a (noisy) signal.
        rng = Random.MersenneTwister(4477)
        T = 200
        x = randn(rng, T)
        y = 0.4 .* x .+ randn(rng, T)
        f_small = fill(mean(y), T)          # restricted
        f_big = 0.4 .* x                    # unrestricted (true signal)
        e_small = y .- f_small
        e_big = y .- f_big
        f_adj = f_small .- f_big
        cw = clark_west(e_small, e_big, f_adj; h=1)
        # The big model genuinely improves → CW should reject (one-sided greater)
        @test cw.statistic > 1.645
        @test cw.pvalue < 0.05
        @test occursin("Clark", sprint(show, cw))
        @test !isempty(refs(cw))
    end

    @testset "Forecast combination" begin
        rng = Random.MersenneTwister(9931)
        T = 300
        truth = cumsum(randn(rng, T)) .* 0.1
        # 3 models with different error variances
        g1 = truth .+ 0.3 .* randn(rng, T)
        g2 = truth .+ 0.6 .* randn(rng, T)
        g3 = truth .+ 1.2 .* randn(rng, T)   # worst
        F = hcat(g1, g2, g3)

        rmse(f) = sqrt(mean((truth .- f).^2))
        indiv_rmse = [rmse(F[:, j]) for j in 1:3]
        worst = maximum(indiv_rmse)

        ceq = combine_forecasts(F, truth; method=:equal)
        @test isapprox(sum(ceq.weights), 1.0; atol=1e-12)
        @test all(ceq.weights .≈ 1/3)
        # PROPERTY: equal-weights beats the worst individual model on RMSE
        @test rmse(ceq.combined) < worst

        cbg = combine_forecasts(F, truth; method=:bates_granger)
        @test isapprox(sum(cbg.weights), 1.0; atol=1e-12)
        @test all(cbg.weights .> 0)
        # inverse-MSE: worst model gets the smallest weight
        @test argmin(cbg.weights) == argmax(cbg.mse)
        @test rmse(cbg.combined) < worst

        cgr = combine_forecasts(F, truth; method=:granger_ramanathan)
        @test isapprox(sum(cgr.weights), 1.0; atol=1e-8)
        # GR minimizes in-sample SSR s.t. sum(w)=1 → its RMSE ≤ equal-weights RMSE
        @test rmse(cgr.combined) <= rmse(ceq.combined) + 1e-8

        @test occursin("Combination", sprint(show, cgr))
        @test !isempty(refs(cgr))
    end

    @testset "plot_result renders" begin
        ev = forecast_evaluate(actual, hcat(f1, f2); model_names=["m1", "m2"])
        p = plot_result(ev; metric="RMSE")
        @test p isa MacroEconometricModels.PlotOutput
    end

    @testset "input validation" begin
        @test_throws DimensionMismatch diebold_mariano(e1, e2[1:end-1])
        @test_throws ArgumentError diebold_mariano(e1, e2; h=0)
        @test_throws ArgumentError combine_forecasts(hcat(f1, f2), actual; method=:bogus)
        @test_throws DimensionMismatch forecast_evaluate(actual, f1[1:end-1])
    end
end
