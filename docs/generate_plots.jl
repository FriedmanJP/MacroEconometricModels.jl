# Generate HTML plot files for documentation embedding
# Run with: julia --project=docs docs/generate_plots.jl
#
# Uses real datasets (FRED-MD, FRED-QD, Penn World Table) when available,
# falling back to synthetic data if loading fails.

using MacroEconometricModels
using DataFrames
using Random

Random.seed!(42)

# Output directory
const PLOT_DIR = joinpath(@__DIR__, "src", "assets", "plots")
mkpath(PLOT_DIR)

function save(name::String, p::PlotOutput)
    path = joinpath(PLOT_DIR, name)
    save_plot(p, path)
    println("  ✓ $name")
end

function _clean_rows(M::Matrix)
    mask = [all(isfinite, M[i,:]) for i in 1:size(M,1)]
    M[mask, :]
end

function main()
    println("Generating documentation plots...")

    # -------------------------------------------------------------------
    # Load real datasets (fallback to synthetic if unavailable)
    # -------------------------------------------------------------------
    use_real = true
    fred_md = nothing; fred_qd = nothing; pwt = nothing
    try
        fred_md = load_example(:fred_md)
        fred_qd = load_example(:fred_qd)
        pwt     = load_example(:pwt)
        println("  ✓ FRED-MD: $(nobs(fred_md)) × $(nvars(fred_md))")
        println("  ✓ FRED-QD: $(nobs(fred_qd)) × $(nvars(fred_qd))")
        println("  ✓ PWT: $(ngroups(pwt)) countries")
    catch e
        @warn "Dataset loading failed, using synthetic data" exception=e
        use_real = false
    end

    # -------------------------------------------------------------------
    # Prepare shared data
    # -------------------------------------------------------------------
    if use_real
        # 3-variable stationary macro panel (VAR / IRF / FEVD / HD / LP)
        key_md = fred_md[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]
        Y3 = _clean_rows(to_matrix(apply_tcode(key_md)))

        # Univariate: INDPRO growth rate (ARIMA)
        y1 = filter(isfinite, apply_tcode(fred_md[:, "INDPRO"], 5))

        # Trending I(1) series: log industrial production (filters)
        y_rw = filter(isfinite, log.(fred_md[:, "INDPRO"]))

        # Volatility: S&P 500 returns (fallback to INDPRO growth)
        y_vol = copy(y1)
        sp_idx = findfirst(v -> occursin("S&P", v) && occursin("500", v),
                           varnames(fred_md))
        if sp_idx !== nothing
            sp_ret = filter(isfinite,
                            apply_tcode(fred_md[:, varnames(fred_md)[sp_idx]], 5))
            length(sp_ret) > 100 && (y_vol = sp_ret)
        end

        # Wide panel for factor models: ≤20 clean transformed series
        X_all = to_matrix(apply_tcode(fred_md))
        good_cols = [j for j in 1:size(X_all,2) if !any(isnan, X_all[:,j])]
        X20 = X_all[:, good_cols[1:min(20, length(good_cols))]]

        # Cointegrated quarterly data for VECM: log GDP components
        qd_sub = fred_qd[:, ["GDPC1", "PCECC96", "GPDIC1"]]
        Y_ci = _clean_rows(log.(to_matrix(qd_sub)))
    else
        Y3    = randn(200, 3)
        y1    = randn(200)
        y_rw  = cumsum(randn(200))
        y_vol = randn(200)
        X20   = randn(200, 20)
        Y_ci  = cumsum(randn(150, 3), dims=1)
    end

    # -------------------------------------------------------------------
    # 1. Quick Start IRF
    # -------------------------------------------------------------------
    m_var = estimate_var(Y3, 2)
    r_qs  = irf(m_var, 20; ci_type=:bootstrap, reps=500)
    save("quickstart_irf.html", plot_result(r_qs))

    # -------------------------------------------------------------------
    # 2. Frequentist IRF (full grid)
    # -------------------------------------------------------------------
    save("irf_freq.html", plot_result(r_qs))

    # -------------------------------------------------------------------
    # 3. Bayesian IRF
    # -------------------------------------------------------------------
    post = estimate_bvar(Y3, 2; n_draws=1000)
    r_birf = irf(post, 20)
    save("irf_bayesian.html", plot_result(r_birf))

    # -------------------------------------------------------------------
    # 4. LP IRF
    # -------------------------------------------------------------------
    lp_m = estimate_lp(Y3, 1, 20; lags=2)
    r_lp = lp_irf(lp_m)
    save("irf_lp.html", plot_result(r_lp))

    # -------------------------------------------------------------------
    # 5. Structural LP IRF
    # -------------------------------------------------------------------
    slp = structural_lp(Y3, 20; method=:cholesky, lags=2)
    save("irf_structural_lp.html", plot_result(slp))

    # -------------------------------------------------------------------
    # 6. Frequentist FEVD
    # -------------------------------------------------------------------
    f_freq = fevd(m_var, 20)
    save("fevd_freq.html", plot_result(f_freq))

    # -------------------------------------------------------------------
    # 7. Bayesian FEVD
    # -------------------------------------------------------------------
    f_bay = fevd(post, 20)
    save("fevd_bayesian.html", plot_result(f_bay))

    # -------------------------------------------------------------------
    # 8. LP-FEVD
    # -------------------------------------------------------------------
    f_lp = lp_fevd(slp, 20)
    save("fevd_lp.html", plot_result(f_lp))

    # -------------------------------------------------------------------
    # 9. Frequentist HD
    # -------------------------------------------------------------------
    hd_freq = historical_decomposition(m_var)
    save("hd_freq.html", plot_result(hd_freq))

    # -------------------------------------------------------------------
    # 10. Bayesian HD
    # -------------------------------------------------------------------
    hd_bay = historical_decomposition(post)
    save("hd_bayesian.html", plot_result(hd_bay))

    # -------------------------------------------------------------------
    # 11-15. Time Series Filters
    # -------------------------------------------------------------------
    save("filter_hp.html",         plot_result(hp_filter(y_rw)))
    save("filter_hamilton.html",   plot_result(hamilton_filter(y_rw); original=y_rw))
    save("filter_bn.html",         plot_result(beveridge_nelson(y_rw)))
    save("filter_bk.html",         plot_result(baxter_king(y_rw); original=y_rw))
    save("filter_boosted_hp.html", plot_result(boosted_hp(y_rw)))

    # -------------------------------------------------------------------
    # 16. ARIMA Forecast
    # -------------------------------------------------------------------
    ar = estimate_ar(y1, 2)
    fc_ar = forecast(ar, 20)
    save("forecast_arima.html", plot_result(fc_ar; history=y1, n_history=30))

    # -------------------------------------------------------------------
    # 17. Volatility Forecast
    # -------------------------------------------------------------------
    gm = estimate_garch(y_vol, 1, 1)
    fc_vol = forecast(gm, 10)
    save("forecast_volatility.html", plot_result(fc_vol; history=gm.conditional_variance))

    # -------------------------------------------------------------------
    # 18. VECM Forecast
    # -------------------------------------------------------------------
    try
        vecm_m  = estimate_vecm(Y_ci, 2; rank=1)
        fc_vecm = forecast(vecm_m, 10)
        save("forecast_vecm.html", plot_result(fc_vecm))
    catch e
        @warn "VECM with real data failed, using synthetic" exception=e
        Y_ci_syn = cumsum(randn(150, 3), dims=1)
        vecm_m   = estimate_vecm(Y_ci_syn, 2; rank=1)
        fc_vecm  = forecast(vecm_m, 10)
        save("forecast_vecm.html", plot_result(fc_vecm))
    end

    # -------------------------------------------------------------------
    # 19. Factor Forecast
    # -------------------------------------------------------------------
    fm = estimate_dynamic_factors(X20, 2, 1)
    fc_fm = forecast(fm, 10)
    save("forecast_factor.html", plot_result(fc_fm))

    # -------------------------------------------------------------------
    # 20. LP Forecast
    # -------------------------------------------------------------------
    Y_lp = Y3[end-99:end, :]
    lp_fc_m = estimate_lp(Y_lp, 1, 10; lags=2)
    shock_path = zeros(10); shock_path[1] = 1.0
    fc_lp = forecast(lp_fc_m, shock_path)
    save("forecast_lp.html", plot_result(fc_lp))

    # -------------------------------------------------------------------
    # 21. GARCH diagnostic
    # -------------------------------------------------------------------
    save("model_garch.html", plot_result(gm))

    # -------------------------------------------------------------------
    # 22. SV posterior volatility
    # -------------------------------------------------------------------
    sv_m = estimate_sv(y_vol; n_samples=2000, burnin=1000)
    save("model_sv.html", plot_result(sv_m))

    # -------------------------------------------------------------------
    # 23. Static factor model
    # -------------------------------------------------------------------
    fm_static = estimate_factors(X20, 3)
    save("model_factor_static.html", plot_result(fm_static))

    # -------------------------------------------------------------------
    # 24. TimeSeriesData
    # -------------------------------------------------------------------
    if use_real
        d_ts = fred_md[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]
    else
        d_ts = TimeSeriesData(randn(100, 3); varnames=["GDP", "CPI", "RATE"])
    end
    save("data_timeseries.html", plot_result(d_ts))

    # -------------------------------------------------------------------
    # 25. PanelData
    # -------------------------------------------------------------------
    if use_real
        save("data_panel.html", plot_result(pwt; vars=["rgdpna", "pop", "emp", "hc"]))
    else
        df = DataFrame(group=repeat(1:3, inner=20), time=repeat(1:20, 3),
                       x=randn(60), y=randn(60))
        pd = xtset(df, :group, :time)
        save("data_panel.html", plot_result(pd))
    end

    # -------------------------------------------------------------------
    # 26. Nowcast result
    # -------------------------------------------------------------------
    if use_real
        nc_md  = fred_md[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
        Y_nc   = _clean_rows(to_matrix(apply_tcode(nc_md)))
        Y_nc   = Y_nc[end-99:end, :]
    else
        Y_nc = randn(100, 5)
    end
    Y_nc[end, end] = NaN
    dfm_nc = nowcast_dfm(Y_nc, 4, 1; r=2, p=1)
    nr = nowcast(dfm_nc)
    save("nowcast_result.html", plot_result(nr))

    # -------------------------------------------------------------------
    # 27. Nowcast news
    # -------------------------------------------------------------------
    X_old = copy(Y_nc)
    X_new = copy(X_old); X_new[end, end] = X_old[end-1, end]
    dfm_news = nowcast_dfm(X_old, 4, 1; r=2, p=1)
    nn = nowcast_news(X_new, X_old, dfm_news, 5)
    save("nowcast_news.html", plot_result(nn))

    println("\nDone! Generated $(length(readdir(PLOT_DIR))) HTML files in $PLOT_DIR")
end

main()
