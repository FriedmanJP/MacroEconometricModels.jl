# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

using Test
using LinearAlgebra
using Statistics
using Distributions
using Random
using StatsAPI
using MacroEconometricModels

if !@isdefined(FAST)
    const FAST = get(ENV, "MACRO_FAST_TESTS", "") == "1"
end

const M_QF = MacroEconometricModels

@testset "QFM Distributional Forecasting & Growth-at-Risk (QFM-07)" begin

    @testset "QFM Single-Quantile Forecast (QAR/QVAR + Standardization + Bootstrap)" begin
        rng = Random.Xoshiro(88901)
        T_obs, N, r = 140, 25, 2
        F_true = zeros(T_obs, r)
        for t in 2:T_obs
            F_true[t, 1] = 0.65 * F_true[t - 1, 1] + 0.10 * F_true[t - 1, 2] + randn(rng)
            F_true[t, 2] = 0.55 * F_true[t - 1, 2] + randn(rng)
        end
        L_true = randn(rng, N, r)
        scales = exp.(range(-0.5, 1.5; length=N))
        shifts = collect(range(-10.0, 10.0; length=N))
        X_raw = (F_true * L_true' .+ 0.3 .* randn(rng, T_obs, N)) .* scales' .+ shifts'

        # Unstandardized QFM forecast
        qfm_raw = estimate_qfm(X_raw, r, 0.5; standardize=false, seed=11)
        fc_none = forecast(qfm_raw, 4; p=1, ci_method=:none)
        @test fc_none isa FactorForecast{Float64}
        @test fc_none.horizon == 4
        @test fc_none.ci_method === :none
        @test size(fc_none.factors) == (4, r)
        @test size(fc_none.observables) == (4, N)
        @test all(==(0.0), fc_none.factors_lower)
        @test all(==(0.0), fc_none.factors_upper)
        @test all(==(0.0), fc_none.observables_lower)
        @test all(==(0.0), fc_none.observables_upper)
        @test all(==(0.0), fc_none.factors_se)
        @test all(==(0.0), fc_none.observables_se)
        @test point_forecast(fc_none) === fc_none.observables
        @test forecast_horizon(fc_none) == 4
        @test fc_none.observables ≈ fc_none.factors * qfm_raw.loadings' atol=1e-12

        # Standardized QFM forecast maps back into raw panel units
        for rob in (true, false)
            qfm_std = estimate_qfm(X_raw, r, 0.5; standardize=true, robust_scale=rob, seed=12)
            fc_std = forecast(qfm_std, 4; p=2, ci_method=:none)
            @test all(isfinite, fc_std.observables)
            # Observable forecasts should lie near original column centers, not standardized ~0
            col_means = vec(mean(X_raw; dims=1))
            @test norm(fc_std.observables[1, :] .- col_means) < norm(col_means)
        end

        # Moving-block bootstrap CIs + seed reproducibility
        n_b = FAST ? 25 : 60
        fc_b1 = forecast(qfm_raw, 4; p=1, ci_method=:bootstrap, conf_level=0.90, n_boot=n_b, seed=77)
        fc_b2 = forecast(qfm_raw, 4; p=1, ci_method=:bootstrap, conf_level=0.90, n_boot=n_b, seed=77)
        @test fc_b1.ci_method === :bootstrap
        @test fc_b1.conf_level ≈ 0.90
        @test fc_b1.factors == fc_b2.factors
        @test fc_b1.observables == fc_b2.observables
        @test fc_b1.observables_lower == fc_b2.observables_lower
        @test fc_b1.observables_upper == fc_b2.observables_upper
        @test all(fc_b1.factors_upper .>= fc_b1.factors_lower)
        @test all(fc_b1.observables_upper .>= fc_b1.observables_lower)
        @test all(>=(0.0), fc_b1.factors_se)
        @test all(>=(0.0), fc_b1.observables_se)
        @test any(>(0.0), fc_b1.observables_se)
        @test fc_b1.observables == fc_none.observables

        # Edge sample / rank fallback stability (T_obs <= k_dim + 1)
        X_tiny = randn(rng, 6, 8)
        qfm_tiny = estimate_qfm(X_tiny, 3, 0.25; n_starts=1, seed=5)
        fc_tiny = forecast(qfm_tiny, 3; p=2, ci_method=:none)
        @test size(fc_tiny.observables) == (3, 8)
        @test all(isfinite, fc_tiny.observables)
    end

    @testset "QuantileFactorGrid Forecast & CFG Monotone Rearrangement (h=1..4)" begin
        rng = Random.Xoshiro(88902)
        T_obs, N = 110, 20
        f = randn(rng, T_obs)
        X = f * randn(rng, N)' .+ 0.8 .* randn(rng, T_obs, N)

        # Include unsorted taus and quantile-varying ranks to verify general rearrangement
        taus_unsorted = [0.75, 0.10, 0.90, 0.25, 0.50]
        grid = estimate_qfm(X, [2, 2, 2, 1, 2], taus_unsorted; standardize=true, seed=21)

        fcs_rearr = forecast(grid, 4; p=1, rearrange=true)
        @test fcs_rearr isa Vector{FactorForecast{Float64}}
        @test length(fcs_rearr) == length(taus_unsorted)

        perm = sortperm(grid.taus)
        for hh in 1:4, i in 1:N
            for j in 1:(length(perm) - 1)
                @test fcs_rearr[perm[j]].observables[hh, i] <=
                      fcs_rearr[perm[j + 1]].observables[hh, i] + 1e-15
            end
        end

        # Also test with bootstrap CIs & rearrange=true
        fcs_boot = forecast(grid, 4; p=1, rearrange=true, ci_method=:bootstrap, n_boot=25, seed=9)
        for hh in 1:4, i in 1:N
            for j in 1:(length(perm) - 1)
                @test fcs_boot[perm[j]].observables[hh, i] <=
                      fcs_boot[perm[j + 1]].observables[hh, i] + 1e-15
                @test fcs_boot[perm[j]].observables_lower[hh, i] <=
                      fcs_boot[perm[j + 1]].observables_lower[hh, i] + 1e-15
                @test fcs_boot[perm[j]].observables_upper[hh, i] <=
                      fcs_boot[perm[j + 1]].observables_upper[hh, i] + 1e-15
            end
        end
    end

    @testset "Azzalini–Capitanio (2003) Skewed-t PDF, CDF, Quantile, ES & Exact Recovery" begin
        # 1. PDF integration to 1 and CDF symmetry: F(z; α, ν) + F(-z; -α, ν) = 1
        for (mu, sig, alp, nu) in ((-0.4, 1.3, -2.2, 3.2), (1.1, 0.7, 1.8, 5.5), (0.0, 1.0, 0.0, 4.0))
            @test M_QF._skewt_cdf(-100.0, mu, sig, alp, nu) < 1e-4
            @test M_QF._skewt_cdf(100.0, mu, sig, alp, nu) > 1.0 - 1e-4
            for z in (-4.5, -1.7, -0.3, 0.0, 0.8, 3.1)
                @test M_QF._skewt_std_cdf(z, alp, nu) + M_QF._skewt_std_cdf(-z, -alp, nu) ≈ 1.0 atol=1e-11
            end
            # α = 0 reduces to exact Student-t
            if alp == 0.0
                for z in (-3.0, -0.5, 1.2)
                    @test M_QF._skewt_cdf(mu + sig * z, mu, sig, 0.0, nu) ≈
                          Distributions.cdf(Distributions.TDist(nu), z) atol=1e-12
                    @test M_QF._skewt_quantile(0.05, mu, sig, 0.0, nu) ≈
                          mu + sig * Distributions.quantile(Distributions.TDist(nu), 0.05) atol=1e-11
                end
            end
            # Inverse CDF inverts CDF accurately
            for p in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
                q_p = M_QF._skewt_quantile(p, mu, sig, alp, nu)
                @test M_QF._skewt_cdf(q_p, mu, sig, alp, nu) ≈ p atol=1e-11
            end
            # Expected Shortfall agrees with direct quantile integral (1/α) ∫_0^α Q(u) du
            # Change of variables u = 0.05 * s^4 removes the u^(-1/ν) tail singularity at u -> 0+
            nodes, weights = M_QF._gauss_legendre_interval(32, 0.0, 1.0)
            es_num = sum(weights[k] * 4.0 * nodes[k]^3 *
                         M_QF._skewt_quantile(max(0.05 * nodes[k]^4, 1e-12), mu, sig, alp, nu)
                         for k in eachindex(nodes))
            es_fast = M_QF._skewt_es(0.05, mu, sig, alp, nu)
            @test es_fast ≈ es_num atol=1e-6
            @test es_fast <= M_QF._skewt_quantile(0.05, mu, sig, alp, nu)
        end

        # 2. Exact quantile matching recovers true (μ, σ, α_skew, ν) within < 1e-4
        anchor_taus = [0.05, 0.25, 0.75, 0.95]
        true_cases = [
            (0.8, 1.4, -2.5, 4.0),
            (-0.5, 0.9, 1.8, 6.0),
            (2.1, 1.1, -1.2, 3.0),
            (0.0, 2.0, 0.4, 8.5),
        ]
        for (mu_t, sig_t, alp_t, nu_t) in true_cases
            q_true = [M_QF._skewt_quantile(p, mu_t, sig_t, alp_t, nu_t) for p in anchor_taus]
            mu_hat, sig_hat, alp_hat, nu_hat = M_QF._skewt_fit_quantiles(q_true, anchor_taus)
            @test abs(mu_hat - mu_t) < 1e-4
            @test abs(sig_hat - sig_t) < 1e-4
            @test abs(alp_hat - alp_t) < 1e-4
            @test abs(nu_hat - nu_t) < 1e-4
        end
    end

    @testset "GrowthAtRisk on Asymmetric Tail-Risk DGP (All Densities, Dispatches & OOS)" begin
        rng = Random.Xoshiro(88903)
        T_obs, N = 180, 30
        f_mean = zeros(T_obs)
        f_vol = zeros(T_obs)
        for t in 2:T_obs
            f_mean[t] = 0.6 * f_mean[t - 1] + 0.8 * randn(rng)
            f_vol[t] = 0.75 * f_vol[t - 1] + 0.6 * randn(rng)
        end
        # Financial stress proxy increases downside tail volatility of y_{t+1}
        stress = clamp.(f_vol, -2.0, 3.5)
        y = zeros(T_obs)
        for t in 2:T_obs
            # Asymmetric left tail: negative shock scales strongly with lagged stress
            sh = randn(rng)
            asym_err = sh < 0 ? (0.4 + 0.9 * max(stress[t - 1], 0.0)) * sh : 0.4 * sh
            y[t] = 0.4 * y[t - 1] + 0.6 * f_mean[t - 1] + asym_err
        end
        X = hcat(y, f_mean * randn(rng, 1, 15) .+ 0.3 * randn(rng, T_obs, 15),
                 stress * abs.(randn(rng, 1, N - 16)) .+ 0.3 * randn(rng, T_obs, N - 16))
        vnames = vcat(["GDP"], ["F$i" for i in 1:(N - 1)])

        # Matrix interface + OOS split + :skewed_t
        taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        gar_oos = growth_at_risk(X, "GDP", 1; r=2, taus=taus, lags=1,
                                 rearrange=true, alpha=0.05, density=:skewed_t,
                                 oos_split=120, varnames=vnames, seed=42)
        @test gar_oos isa GrowthAtRiskResult{Float64}
        @test gar_oos.target_name == "GDP"
        @test gar_oos.horizon == 1
        @test gar_oos.lags == 1
        @test gar_oos.density === :skewed_t
        @test gar_oos.rearrange === true
        @test gar_oos.n_obs == T_obs
        @test gar_oos.n_eff == T_obs - 1
        @test StatsAPI.nobs(gar_oos) == T_obs
        @test M_QF.effective_nobs(gar_oos) == T_obs - 1
        @test size(gar_oos.fitted_quantiles) == (T_obs - 1, length(taus))
        @test size(gar_oos.density_params) == (T_obs - 1, 4)
        @test length(gar_oos.oos_density_params) == 4
        @test all(>(0.0), gar_oos.density_params[:, 2])
        @test all(>(2.05), gar_oos.density_params[:, 4])

        # Monotone rearrangement invariant across all rows and OOS forecast
        for t in 1:gar_oos.n_eff, j in 1:(length(taus) - 1)
            @test gar_oos.fitted_quantiles[t, j] <= gar_oos.fitted_quantiles[t, j + 1] + 1e-15
        end
        for j in 1:(length(taus) - 1)
            @test gar_oos.oos_quantiles[j] <= gar_oos.oos_quantiles[j + 1] + 1e-15
        end

        # Tail risk & scoring invariants
        @test all(isfinite, gar_oos.gar)
        @test all(isfinite, gar_oos.expected_shortfall)
        @test all(gar_oos.expected_shortfall .<= gar_oos.gar .+ 1e-8)
        @test gar_oos.oos_expected_shortfall <= gar_oos.oos_gar + 1e-8
        @test all(isfinite, gar_oos.upside_entropy)
        @test all(isfinite, gar_oos.downside_entropy)
        # Total relative entropy KL(f || g) = downside + upside >= 0
        @test all(>=(-1e-6), gar_oos.upside_entropy .+ gar_oos.downside_entropy)
        @test gar_oos.oos_upside_entropy + gar_oos.oos_downside_entropy >= -1e-6
        @test length(gar_oos.quantile_score) == length(taus)
        @test all(>=(0.0), gar_oos.quantile_score)
        @test length(gar_oos.crps) == gar_oos.n_eff
        @test all(>=(0.0), gar_oos.crps)

        # OOS evaluation vectors populated when oos_split !== nothing
        n_oos = (T_obs - 1) - 120 + 1
        @test length(gar_oos.pit) == n_oos
        @test all(p -> 0.0 <= p <= 1.0, gar_oos.pit)
        @test length(gar_oos.oos_crps) == n_oos
        @test all(>=(0.0), gar_oos.oos_crps)
        @test length(gar_oos.oos_quantile_scores) == length(taus)
        @test all(>=(0.0), gar_oos.oos_quantile_scores)

        # When oos_split === nothing, OOS vectors are empty Float64[]
        gar_in = growth_at_risk(X, 1, 1; r=2, taus=taus, lags=1, seed=42)
        @test isempty(gar_in.pit) && gar_in.pit isa Vector{Float64}
        @test isempty(gar_in.oos_quantile_scores) && gar_in.oos_quantile_scores isa Vector{Float64}
        @test isempty(gar_in.oos_crps) && gar_in.oos_crps isa Vector{Float64}
        @test gar_in.oos_split === nothing
        @test gar_in.gar == gar_oos.gar
        @test gar_in.oos_quantiles == gar_oos.oos_quantiles

        # Grid dispatch across Int, String, Symbol, and external Vector targets
        grid = estimate_qfm(X, 2, taus; varnames=vnames, standardize=true, seed=55)
        g_int = growth_at_risk(grid, 1, 1; seed=1)
        g_str = growth_at_risk(grid, "GDP", 1; seed=1)
        g_sym = growth_at_risk(grid, :GDP, 1; seed=1)
        g_vec = growth_at_risk(grid, y, 1; seed=1)
        @test g_int.gar == g_str.gar == g_sym.gar == g_vec.gar
        @test g_int.oos_quantiles == g_vec.oos_quantiles

        # Custom taus subset on grid & h=4
        g_h4 = growth_at_risk(grid, 1, 4; taus=[0.05, 0.25, 0.75, 0.95], lags=2, seed=2)
        @test g_h4.horizon == 4 && g_h4.lags == 2
        @test g_h4.n_eff == T_obs - 4 - 2 + 1
        for t in 1:g_h4.n_eff, j in 1:3
            @test g_h4.fitted_quantiles[t, j] <= g_h4.fitted_quantiles[t, j + 1] + 1e-15
        end

        # Normal and :none density options (including lags=0 and OOS)
        g_norm = growth_at_risk(grid, 1, 1; density=:normal, oos_split=140, lags=0)
        @test g_norm.density === :normal
        @test size(g_norm.density_params) == (T_obs - 1, 2)
        @test length(g_norm.oos_density_params) == 2
        @test all(g_norm.expected_shortfall .<= g_norm.gar .+ 1e-10)
        @test all(>=(-1e-10), g_norm.upside_entropy .+ g_norm.downside_entropy)
        @test length(g_norm.pit) == (T_obs - 1) - 140 + 1

        g_emp = growth_at_risk(grid, 1, 1; density=:none, oos_split=140)
        @test g_emp.density === :none
        @test size(g_emp.density_params) == (0, 0)
        @test isempty(g_emp.oos_density_params)
        @test all(isfinite, g_emp.gar)
        @test all(g_emp.expected_shortfall .<= g_emp.gar .+ 1e-8)
        @test all(isnan, g_emp.upside_entropy)
        @test all(isnan, g_emp.downside_entropy)
        @test all(>=(0.0), g_emp.crps)
        @test length(g_emp.pit) == (T_obs - 1) - 140 + 1
    end

    @testset "End-to-End :fred_qd Subset (< 2s & Asymmetric Left-Tail Widening)" begin
        qd = load_example(:fred_qd)
        vars = ["GDPC1", "PCECC96", "GPDIC1", "INDPRO", "PAYEMS", "UNRATE",
                "HOUST", "CPIAUCSL", "FEDFUNDS", "TB3MS", "GS1", "GS10",
                "AAA", "BAA", "BAA10YM", "GS10TB3Mx", "M2REAL", "S&P 500"]
        qd_sub = apply_tcode(qd[:, vars])
        X_raw = to_matrix(qd_sub)
        keep = [all(isfinite, @view(X_raw[t, :])) for t in 1:size(X_raw, 1)]
        X_clean = Matrix{Float64}(X_raw[keep, :])
        # Annualized quarterly GDP growth (%) in column 1
        X_clean[:, 1] .*= 400.0

        # Warmup JIT briefly on tiny slice, then time the full :fred_qd subset run
        _ = growth_at_risk(X_clean[1:40, 1:6], 1, 1; r=2, lags=1, alpha=0.05,
                           density=:skewed_t, varnames=vars[1:6], seed=42)
        local gar_qd
        t_elapsed = @elapsed begin
            gar_qd = growth_at_risk(X_clean, 1, 1; r=2, lags=1, alpha=0.05,
                                    density=:skewed_t, varnames=vars, seed=42)
        end
        @test t_elapsed < 20.0
        @test gar_qd isa GrowthAtRiskResult{Float64}
        @test gar_qd.target_name == "GDPC1"
        @test all(isfinite, gar_qd.gar)
        @test all(gar_qd.expected_shortfall .<= gar_qd.gar .+ 1e-8)

        # Asymmetric left-tail widening vs upper-tail stability (ABG 2019 stylized fact)
        idx_05 = findfirst(==(0.05), gar_qd.taus)
        idx_50 = findfirst(==(0.50), gar_qd.taus)
        idx_95 = findfirst(==(0.95), gar_qd.taus)
        q05 = @view gar_qd.fitted_quantiles[:, idx_05]
        q50 = @view gar_qd.fitted_quantiles[:, idx_50]
        q95 = @view gar_qd.fitted_quantiles[:, idx_95]
        # Lower tail fluctuates substantially more than the upper tail
        @test std(q05) > std(q95)
        @test (maximum(q50 .- q05) - minimum(q50 .- q05)) >
              (maximum(q95 .- q50) - minimum(q95 .- q50))
    end

    @testset "Float32, Int Fallback, RSER-01 _generic_construct, Repro, Display & Validation" begin
        rng = Random.Xoshiro(88904)
        T_obs, N = 90, 12
        X = randn(rng, T_obs, N)
        X32 = Float32.(X)

        qfm32 = estimate_qfm(X32, 2, 0.5f0; seed=32)
        fc32 = forecast(qfm32, 3; ci_method=:bootstrap, n_boot=15, seed=32)
        @test fc32 isa FactorForecast{Float32}
        @test eltype(fc32.observables) === Float32
        @test eltype(fc32.observables_lower) === Float32

        grid32 = estimate_qfm(X32, 2, Float32[0.05, 0.25, 0.50, 0.75, 0.95]; seed=33)
        fcs32 = forecast(grid32, 3; rearrange=true)
        @test fcs32 isa Vector{FactorForecast{Float32}}

        gar32 = growth_at_risk(X32, 1, 1; r=2, oos_split=70, seed=34)
        @test gar32 isa GrowthAtRiskResult{Float32}
        @test eltype(gar32.gar) === Float32
        @test eltype(gar32.pit) === Float32
        @test eltype(gar32.oos_crps) === Float32
        @test eltype(gar32.oos_quantile_scores) === Float32

        # Integer input promotion
        X_int = round.(Int, 10 .* X)
        gar_int = growth_at_risk(X_int, 1, 1; r=2, seed=35)
        @test gar_int isa GrowthAtRiskResult{Float64}

        # Seed reproducibility & manifest presence/absence
        gar_s1 = growth_at_risk(X, 1, 1; r=2, oos_split=70, seed=123)
        gar_s2 = growth_at_risk(X, 1, 1; r=2, oos_split=70, seed=123)
        @test gar_s1.manifest isa ReproManifest
        @test gar_s1.manifest.seed == 123
        @test gar_s1.gar == gar_s2.gar
        @test gar_s1.oos_quantiles == gar_s2.oos_quantiles
        @test gar_s1.pit == gar_s2.pit
        @test gar_s1.oos_crps == gar_s2.oos_crps

        gar_noseed = growth_at_risk(X, 1, 1; r=2, seed=nothing)
        @test gar_noseed.manifest === nothing

        # RSER-01 positional constructor & _generic_construct in exact fieldnames order
        fvals = Any[getfield(gar_s1, f) for f in fieldnames(GrowthAtRiskResult)]
        gar_pos = GrowthAtRiskResult{Float64}(fvals...)
        gar_gen = M_QF._generic_construct(GrowthAtRiskResult, fvals)
        @test gar_pos.gar == gar_s1.gar
        @test gar_pos.oos_quantiles == gar_s1.oos_quantiles
        @test gar_pos.manifest == gar_s1.manifest
        @test gar_gen isa GrowthAtRiskResult{Float64}
        @test gar_gen.gar == gar_s1.gar
        @test gar_gen.pit == gar_s1.pit

        # Display across backends & repro=true
        for obj in (gar_s1, gar_noseed)
            for be in (:text, :latex, :html)
                M_QF.with_display_backend(be) do
                    @test !isempty(sprint(show, obj))
                    @test !isempty(sprint(report, obj))
                end
            end
        end
        @test occursin("Reproducibility: seed=123", sprint(io -> report(io, gar_s1; repro=true)))
        @test !occursin("Reproducibility:", sprint(io -> report(io, gar_s1; repro=false)))

        # ArgumentError & DimensionMismatch checks
        @test_throws ArgumentError forecast(qfm32, 0)
        @test_throws ArgumentError forecast(qfm32, 2; p=0)
        @test_throws ArgumentError forecast(qfm32, 2; ci_method=:theoretical)
        @test_throws ArgumentError forecast(qfm32, 2; conf_level=1.0)
        @test_throws ArgumentError forecast(qfm32, 2; ci_method=:bootstrap, n_boot=1)
        @test_throws ArgumentError forecast(grid32, 0)

        @test_throws ArgumentError growth_at_risk(X, 1, 0)
        @test_throws ArgumentError growth_at_risk(X, 1, 1; lags=-1)
        @test_throws ArgumentError growth_at_risk(X, 1, 1; alpha=0.0)
        @test_throws ArgumentError growth_at_risk(X, 1, 1; alpha=1.0)
        @test_throws ArgumentError growth_at_risk(X, 1, 1; density=:bogus)
        @test_throws ArgumentError growth_at_risk(X, 1, 1; taus=[0.1, 0.5, 0.9], density=:skewed_t)
        @test_throws ArgumentError growth_at_risk(X, 1, 1; taus=[0.5], density=:normal)
        @test_throws ArgumentError growth_at_risk(X, 1, 1; taus=[0.2, 0.2, 0.5, 0.8])
        @test_throws ArgumentError growth_at_risk(X, 1, 1; taus=Float64[])
        @test_throws ArgumentError growth_at_risk(X, 0, 1)
        @test_throws ArgumentError growth_at_risk(X, N + 1, 1)
        @test_throws ArgumentError growth_at_risk(X, "GDP", 1)
        @test_throws ArgumentError growth_at_risk(X, "NONEXISTENT", 1; varnames=["V$i" for i in 1:N])
        y_bad = randn(rng, T_obs - 1)
        @test_throws ArgumentError growth_at_risk(X, y_bad, 1)
        @test_throws ArgumentError growth_at_risk(X, 1, 1; oos_split=2)
        @test_throws ArgumentError growth_at_risk(X, 1, 1; oos_split=T_obs)
        X_bad = randn(rng, T_obs - 2, N)
        @test_throws DimensionMismatch growth_at_risk(grid32, 1, 1; X=X_bad)
    end
end
