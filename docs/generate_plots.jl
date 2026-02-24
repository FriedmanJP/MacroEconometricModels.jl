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
        # Subset to Great Moderation (1987:01–2007:12) for cleaner macro data
        gm_start = (1987 - 1959) * 12 + 1  # row 337
        gm_end   = min((2007 - 1959) * 12 + 12, nobs(fred_md))  # row 588
        fred_gm = TimeSeriesData(fred_md.data[gm_start:gm_end, :];
                     varnames=fred_md.varnames, frequency=fred_md.frequency,
                     tcode=fred_md.tcode)
        println("  ✓ Great Moderation subset: $(nobs(fred_gm)) monthly obs")

        # 3-variable stationary macro panel (VAR / IRF / FEVD / HD / LP)
        key_md = fred_gm[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]
        Y3 = _clean_rows(to_matrix(apply_tcode(key_md)))

        # Trending I(1) series: log INDPRO (for filters + ARIMA(1,1,0) with widening CIs)
        y_rw = filter(isfinite, log.(fred_gm[:, "INDPRO"]))

        # Volatility: INDPRO growth rate (fallback to S&P 500 if available)
        y_vol = filter(isfinite, apply_tcode(fred_gm[:, "INDPRO"], 5))
        sp_idx = findfirst(v -> occursin("S&P", v) && occursin("500", v),
                           varnames(fred_gm))
        if sp_idx !== nothing
            sp_ret = filter(isfinite,
                            apply_tcode(fred_gm[:, varnames(fred_gm)[sp_idx]], 5))
            length(sp_ret) > 100 && (y_vol = sp_ret)
        end

        # Wide panel for factor models: ≤20 clean transformed series
        safe_idx = [i for i in 1:nvars(fred_gm)
                    if fred_gm.tcode[i] < 4 || all(x -> isfinite(x) && x > 0, fred_gm.data[:, i])]
        fred_safe = fred_gm[:, varnames(fred_gm)[safe_idx]]
        X_all = to_matrix(apply_tcode(fred_safe))
        good_cols = [j for j in 1:size(X_all,2) if !any(isnan, X_all[:,j])]
        X20 = X_all[:, good_cols[1:min(20, length(good_cols))]]

        # Cointegrated quarterly data for VECM: log GDP components (Great Moderation)
        gm_q_start = (1987 - 1959) * 4 + 1  # row 113
        gm_q_end   = min((2007 - 1959) * 4 + 4, nobs(fred_qd))  # row 196
        fred_qd_gm = TimeSeriesData(fred_qd.data[gm_q_start:gm_q_end, :];
                       varnames=fred_qd.varnames, frequency=fred_qd.frequency,
                       tcode=fred_qd.tcode)
        qd_sub = fred_qd_gm[:, ["GDPC1", "PCECC96", "GPDIC1"]]
        Y_ci = _clean_rows(log.(to_matrix(qd_sub)))
    else
        Y3    = randn(200, 3)
        y_rw  = cumsum(0.002 .+ 0.01 .* randn(200))
        y_vol = randn(200)
        X20   = randn(200, 20)
        Y_ci  = cumsum(randn(150, 3), dims=1)
    end

    # -------------------------------------------------------------------
    # 1. Quick Start IRF
    # -------------------------------------------------------------------
    m_var = estimate_var(Y3, 4; varnames=["INDPRO", "UNRATE", "CPI"])
    r_qs  = irf(m_var, 20; ci_type=:bootstrap, reps=500)
    save("quickstart_irf.html", plot_result(r_qs))

    # -------------------------------------------------------------------
    # 2. Frequentist IRF (full grid)
    # -------------------------------------------------------------------
    save("irf_freq.html", plot_result(r_qs))

    # -------------------------------------------------------------------
    # 3. Bayesian IRF
    # -------------------------------------------------------------------
    post = estimate_bvar(Y3, 4; n_draws=1000, varnames=["INDPRO", "UNRATE", "CPI"])
    r_birf = irf(post, 20)
    save("irf_bayesian.html", plot_result(r_birf))

    # -------------------------------------------------------------------
    # 4. LP IRF
    # -------------------------------------------------------------------
    lp_m = estimate_lp(Y3, 1, 20; lags=4, varnames=["INDPRO", "UNRATE", "CPI"])
    r_lp = lp_irf(lp_m)
    save("irf_lp.html", plot_result(r_lp))

    # -------------------------------------------------------------------
    # 5. Structural LP IRF
    # -------------------------------------------------------------------
    slp = structural_lp(Y3, 20; method=:cholesky, lags=4, varnames=["INDPRO", "UNRATE", "CPI"])
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
    ar = estimate_arima(y_rw, 1, 1, 0)
    fc_ar = forecast(ar, 20)
    save("forecast_arima.html", plot_result(fc_ar; history=y_rw, n_history=30))

    # -------------------------------------------------------------------
    # 17. Volatility Forecast
    # -------------------------------------------------------------------
    gm = estimate_garch(y_vol, 1, 1)
    fc_vol = forecast(gm, 10)
    save("forecast_volatility.html", plot_result(fc_vol; history=gm.conditional_variance))

    # -------------------------------------------------------------------
    # 18. VAR Forecast
    # -------------------------------------------------------------------
    # Use same 3-variable macro data; bootstrap CIs show forecast uncertainty
    fc_var = forecast(m_var, 12; ci_method=:bootstrap, reps=500)
    save("forecast_var.html", plot_result(fc_var))

    # -------------------------------------------------------------------
    # 19. BVAR Forecast
    # -------------------------------------------------------------------
    fc_bvar = forecast(post, 12)
    save("forecast_bvar.html", plot_result(fc_bvar))

    # -------------------------------------------------------------------
    # 20. VECM Forecast
    # -------------------------------------------------------------------
    try
        vecm_m  = estimate_vecm(Y_ci, 2; rank=1, varnames=["GDP", "PCE", "Investment"])
        fc_vecm = forecast(vecm_m, 10)
        save("forecast_vecm.html", plot_result(fc_vecm))
    catch e
        @warn "VECM with real data failed, using synthetic" exception=e
        Y_ci_syn = cumsum(randn(150, 3), dims=1)
        vecm_m   = estimate_vecm(Y_ci_syn, 2; rank=1, varnames=["GDP", "PCE", "Investment"])
        fc_vecm  = forecast(vecm_m, 10)
        save("forecast_vecm.html", plot_result(fc_vecm))
    end

    # -------------------------------------------------------------------
    # 21. Factor Forecast
    # -------------------------------------------------------------------
    # Use raw-level FRED-MD interest rate/spread panel (no tcode transform)
    # for visually meaningful factor forecasts with non-zero magnitudes
    _fac_vars = ["FEDFUNDS", "GS1", "GS10", "AAA", "BAA",
                 "T5YFFM", "T10YFFM", "AAAFFM", "BAAFFM"]
    _fac_avail = [v for v in _fac_vars if v in varnames(fred_gm)]
    _X_fac = to_matrix(fred_gm[:, _fac_avail])
    _fac_good = [j for j in 1:size(_X_fac,2) if !any(isnan, _X_fac[:,j])]
    _X_fac = _X_fac[:, _fac_good]
    fm = estimate_dynamic_factors(_X_fac, 3, 2; standardize=true)
    fc_fm = forecast(fm, 12)
    save("forecast_factor.html", plot_result(fc_fm; n_obs=4))

    # -------------------------------------------------------------------
    # 22. LP Forecast
    # -------------------------------------------------------------------
    # Use raw-level FRED-MD macro data (no tcode transform) for meaningful
    # LP forecasts: 1pp fed funds rate shock → unemployment + 10y yield
    _lp_vars = ["FEDFUNDS", "UNRATE", "GS10"]
    _lp_avail = [v for v in _lp_vars if v in varnames(fred_gm)]
    _Y_lp = to_matrix(fred_gm[:, _lp_avail])
    _lp_good = [i for i in 1:size(_Y_lp,1) if all(isfinite, _Y_lp[i,:])]
    _Y_lp = _Y_lp[_lp_good, :]
    _Y_lp = _Y_lp[end-99:end, :]
    lp_fc_m = estimate_lp(_Y_lp, 1, 10; lags=4, varnames=_lp_avail)
    shock_path = zeros(10); shock_path[1] = 1.0
    fc_lp = forecast(lp_fc_m, shock_path)
    save("forecast_lp.html", plot_result(fc_lp))

    # -------------------------------------------------------------------
    # 23. GARCH diagnostic
    # -------------------------------------------------------------------
    save("model_garch.html", plot_result(gm))

    # -------------------------------------------------------------------
    # 24. SV posterior volatility
    # -------------------------------------------------------------------
    sv_m = estimate_sv(y_vol; n_samples=500, burnin=200)
    save("model_sv.html", plot_result(sv_m))

    # -------------------------------------------------------------------
    # 25. Static factor model
    # -------------------------------------------------------------------
    fm_static = estimate_factors(X20, 3)
    save("model_factor_static.html", plot_result(fm_static))

    # -------------------------------------------------------------------
    # 26. TimeSeriesData
    # -------------------------------------------------------------------
    if use_real
        d_ts = fred_gm[:, ["INDPRO", "UNRATE", "CPIAUCSL"]]
    else
        d_ts = TimeSeriesData(randn(100, 3); varnames=["GDP", "CPI", "RATE"])
    end
    save("data_timeseries.html", plot_result(d_ts))

    # -------------------------------------------------------------------
    # 27. PanelData
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
    # 28. Nowcast result
    # -------------------------------------------------------------------
    if use_real
        nc_md  = fred_gm[:, ["INDPRO", "UNRATE", "CPIAUCSL", "M2SL", "FEDFUNDS"]]
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
    # 29. Nowcast news
    # -------------------------------------------------------------------
    X_old = copy(Y_nc)
    X_new = copy(X_old); X_new[end, end] = X_old[end-1, end]
    dfm_news = nowcast_dfm(X_old, 4, 1; r=2, p=1)
    nn = nowcast_news(X_new, X_old, dfm_news, 5)
    save("nowcast_news.html", plot_result(nn))

    # -------------------------------------------------------------------
    # 30. DSGE IRF
    # -------------------------------------------------------------------
    dsge_spec = @dsge begin
        parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.01
        endogenous: Y, C, K, A
        exogenous: ε_A

        Y[t] = A[t] * K[t-1]^α
        C[t] + K[t] = Y[t] + (1 - δ) * K[t-1]
        1 = β * (C[t] / C[t+1]) * (α * A[t+1] * K[t]^(α - 1) + 1 - δ)
        A[t] = ρ * A[t-1] + σ * ε_A[t]

        steady_state = begin
            A_ss = 1.0
            K_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
            Y_ss = K_ss^α
            C_ss = Y_ss - δ * K_ss
            [Y_ss, C_ss, K_ss, A_ss]
        end
    end
    dsge_sol = solve(dsge_spec)
    dsge_irf = irf(dsge_sol, 40)
    save("dsge_irf.html", plot_result(dsge_irf))

    # -------------------------------------------------------------------
    # 31. DSGE FEVD (NK model with demand + supply shocks)
    # -------------------------------------------------------------------
    nk_fevd_spec = @dsge begin
        parameters: β = 0.99, σ_c = 1.0, κ = 0.3, φ_π = 1.5, φ_y = 0.5,
                    ρ_d = 0.8, ρ_s = 0.7, σ_d = 0.01, σ_s = 0.01
        endogenous: y, π, R, d, s
        exogenous: ε_d, ε_s

        y[t] = y[t+1] - (1 / σ_c) * (R[t] - π[t+1]) + d[t]
        π[t] = β * π[t+1] + κ * y[t] + s[t]
        R[t] = φ_π * π[t] + φ_y * y[t]
        d[t] = ρ_d * d[t-1] + σ_d * ε_d[t]
        s[t] = ρ_s * s[t-1] + σ_s * ε_s[t]
    end
    nk_fevd_sol = solve(nk_fevd_spec)
    dsge_fevd = fevd(nk_fevd_sol, 40)
    save("dsge_fevd.html", plot_result(dsge_fevd))

    # -------------------------------------------------------------------
    # 32. OccBin IRF comparison (large negative demand shock hits ZLB)
    # -------------------------------------------------------------------
    nk_occ_spec = @dsge begin
        parameters: β = 0.99, σ_c = 1.0, κ = 0.3, φ_π = 1.5, φ_y = 0.5,
                    ρ_d = 0.8, σ_d = 0.01
        endogenous: y, π, R, d
        exogenous: ε_d

        y[t] = y[t+1] - (1 / σ_c) * (R[t] - π[t+1]) + d[t]
        π[t] = β * π[t+1] + κ * y[t]
        R[t] = φ_π * π[t] + φ_y * y[t]
        d[t] = ρ_d * d[t-1] + σ_d * ε_d[t]
    end
    nk_occ_constraint = parse_constraint(:(R[t] >= 0), nk_occ_spec)
    oirf = occbin_irf(nk_occ_spec, nk_occ_constraint, 1, 80; magnitude=8.0)
    save("occbin_irf.html", plot_result(oirf))

    # -------------------------------------------------------------------
    # 33. OccBin solution path
    # -------------------------------------------------------------------
    occ_shocks = zeros(80, 1)
    occ_shocks[1, 1] = -8.0
    occ_sol = occbin_solve(nk_occ_spec, nk_occ_constraint; shock_path=occ_shocks)
    save("occbin_solution.html", plot_result(occ_sol))

    println("\nDone! Generated $(length(readdir(PLOT_DIR))) HTML files in $PLOT_DIR")
end

main()
