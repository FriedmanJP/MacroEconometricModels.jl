# Generate HTML plot files for documentation embedding
# Run with: julia --project=docs docs/generate_plots.jl
#
# Uses real datasets (FRED-MD, FRED-QD, Penn World Table) when available,
# falling back to synthetic data if loading fails.
#
# Assets are COMMITTED to the repo (docs/src/assets/plots/) and regenerated
# whenever plotting code changes. Every emitted asset is embedded by some docs
# page (mostly the Visualization gallery in docs/src/plotting.md); orphan assets
# are deleted. New/experimental dispatches are wrapped in `try_save` so a single
# estimator failure never aborts the whole generator (mirrors the `use_real`
# dataset fallback). The generator doubles as a plotting smoke test — it exercises
# dispatches that unit tests may not reach.

using MacroEconometricModels
using DataFrames
using Distributions
using Random
using LinearAlgebra
using SparseArrays

const MEM = MacroEconometricModels

Random.seed!(42)

# Output directory
const PLOT_DIR = joinpath(@__DIR__, "src", "assets", "plots")
mkpath(PLOT_DIR)

function save(name::String, p::PlotOutput)
    path = joinpath(PLOT_DIR, name)
    save_plot(p, path)
    println("  ✓ $name")
end

# Guarded save: build the plot inside `thunk` and write it under `name`; on any
# failure, warn and continue so the generator finishes the remaining assets.
function try_save(thunk, name::String)
    try
        save(name, thunk())
    catch e
        @warn "Skipped $name" exception=(e, catch_backtrace())
    end
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
        # Must match doc @setup blocks: ["INDPRO", "CPIAUCSL", "FEDFUNDS"]
        key_md = fred_gm[:, ["INDPRO", "CPIAUCSL", "FEDFUNDS"]]
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
    m_var = estimate_var(Y3, 4; varnames=["INDPRO", "CPI", "FFR"])
    r_qs  = irf(m_var, 20; ci_type=:bootstrap, reps=500)
    save("quickstart_irf.html", plot_result(r_qs))

    # -------------------------------------------------------------------
    # 2. Frequentist IRF (full grid)
    # -------------------------------------------------------------------
    save("irf_freq.html", plot_result(r_qs))

    # -------------------------------------------------------------------
    # 2b. Sign-restricted IRF
    # -------------------------------------------------------------------
    check_fn = (irfs) -> irfs[1, 1, 1] > 0 && irfs[1, 2, 1] < 0
    r_sign = irf(m_var, 20; method=:sign, check_func=check_fn)
    save("irf_sign.html", plot_result(r_sign))

    # -------------------------------------------------------------------
    # 2c. Long-run restricted IRF
    # -------------------------------------------------------------------
    r_lr = irf(m_var, 40; method=:long_run)
    save("irf_longrun.html", plot_result(r_lr))

    # -------------------------------------------------------------------
    # 3. Bayesian IRF
    # -------------------------------------------------------------------
    post = estimate_bvar(Y3, 4; n_draws=1000, varnames=["INDPRO", "CPI", "FFR"])
    r_birf = irf(post, 20)
    save("irf_bayesian.html", plot_result(r_birf))

    # -------------------------------------------------------------------
    # 4. LP IRF
    # -------------------------------------------------------------------
    lp_m = estimate_lp(Y3, 1, 20; lags=4, varnames=["INDPRO", "CPI", "FFR"])
    r_lp = lp_irf(lp_m)
    save("irf_lp.html", plot_result(r_lp))

    # -------------------------------------------------------------------
    # 5. Structural LP IRF
    # -------------------------------------------------------------------
    slp = structural_lp(Y3, 20; method=:cholesky, lags=4, varnames=["INDPRO", "CPI", "FFR"])
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

    # X-13ARIMA-SEATS decomposition (Box-Jenkins airline passengers, 1949-1960)
    airline = Float64[
        112,118,132,129,121,135,148,148,136,119,104,118,
        115,126,141,135,125,149,170,170,158,133,114,140,
        145,150,178,163,172,178,199,199,184,162,146,166,
        171,180,193,181,183,218,230,242,209,191,172,194,
        196,196,236,235,229,243,264,272,237,211,180,201,
        204,188,235,227,234,264,302,293,259,229,203,229,
        242,233,267,269,270,315,364,347,312,274,237,278,
        284,277,317,313,318,374,413,405,355,306,271,306,
        315,301,356,348,355,422,465,467,404,347,305,336,
        340,318,362,348,363,435,491,505,404,359,310,337,
        360,342,406,396,420,472,548,559,463,407,362,405,
        417,391,419,461,472,535,622,606,508,461,390,432]
    save("x13_decomp.html", plot_result(x13_filter(airline; frequency=12, method=:x11);
         title="X-13ARIMA-SEATS — Box-Jenkins Airline Passengers (1949–1960)"))

    # -------------------------------------------------------------------
    # 16. ARIMA Forecast (matches arima.md @setup: CPI log differences)
    # -------------------------------------------------------------------
    y_cpi = filter(isfinite, diff(log.(fred_gm[:, "CPIAUCSL"])))
    arima_m = estimate_arima(y_cpi, 1, 1, 0)
    fc_arima = forecast(arima_m, 20)
    save("forecast_arima.html", plot_result(fc_arima; history=y_cpi, n_history=30))

    # -------------------------------------------------------------------
    # 16b. AR Forecast (stationary CPI inflation, matches arima.md Recipe 6)
    # -------------------------------------------------------------------
    ar_pure = estimate_ar(y_cpi, 2)
    fc_ar_pure = forecast(ar_pure, 20)
    save("forecast_ar.html", plot_result(fc_ar_pure; history=y_cpi, n_history=30))

    # -------------------------------------------------------------------
    # 16c. ARMA Forecast (stationary CPI inflation)
    # -------------------------------------------------------------------
    arma_m = estimate_arma(y_cpi, 2, 1)
    fc_arma = forecast(arma_m, 12)
    save("forecast_arma.html", plot_result(fc_arma; history=y_cpi, n_history=30))

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
    # 20. VECM Forecast + IRF + FEVD
    # -------------------------------------------------------------------
    local vecm_m
    try
        vecm_m  = estimate_vecm(Y_ci, 2; rank=1, varnames=["GDP", "PCE", "Investment"])
    catch e
        @warn "VECM with real data failed, using synthetic" exception=e
        Y_ci_syn = cumsum(randn(150, 3), dims=1)
        vecm_m   = estimate_vecm(Y_ci_syn, 2; rank=1, varnames=["GDP", "PCE", "Investment"])
    end
    fc_vecm = forecast(vecm_m, 10)
    save("forecast_vecm.html", plot_result(fc_vecm))
    vecm_irf_r = irf(vecm_m, 20; method=:cholesky)
    save("vecm_irf.html", plot_result(vecm_irf_r))
    vecm_fevd_r = fevd(vecm_m, 20)
    save("vecm_fevd.html", plot_result(vecm_fevd_r))

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
    save("nowcast_heatmap.html", plot_result(nr; view=:heatmap,
         variable_names=["INDPRO", "UNRATE", "CPI", "M2", "FFR"]))
    save("nowcast_contributions.html", plot_result(nr; view=:contributions))

    # -------------------------------------------------------------------
    # 29. Nowcast news
    # -------------------------------------------------------------------
    X_old = copy(Y_nc)
    X_new = copy(X_old); X_new[end, end] = X_old[end-1, end]
    dfm_news = nowcast_dfm(X_old, 4, 1; r=2, p=1)
    nn = nowcast_news(X_new, X_old, dfm_news, 5)
    save("nowcast_news.html", plot_result(nn))

    X_old2 = copy(Y_nc)
    X_old2[end, 1:3] .= NaN
    groups = [1, 1, 2, 2, 2]
    nn_grp = nowcast_news(Y_nc, X_old2, dfm_nc, size(Y_nc, 1);
             target_var=5, groups=groups,
             group_names=["Real", "Nominal"])
    save("nowcast_news_groups.html", plot_result(nn_grp; view=:groups))
    save("nowcast_news_individual.html", plot_result(nn_grp; view=:individual))

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
    # 32. OccBin IRF comparison (borrowing constraint)
    # -------------------------------------------------------------------
    borrow_spec = @dsge begin
        parameters: β = 20/21, R = 21/20, ρ = 0.9, σ = 0.05, M = 1.0
        endogenous: b, c, y
        exogenous: u

        b[t] = (y[t+1] + b[t+1] + R * b[t-1] - y[t]) / (1 + R)
        c[t] = y[t] + b[t] - R * b[t-1]
        y[t] = y[t-1]^ρ * exp(σ * u[t])
    end
    borrow_spec = compute_steady_state(borrow_spec;
        method=:analytical, ss_fn = θ -> [0.0, 1.0, 1.0])
    borrow_constraint = parse_constraint(:(b[t] <= 1.0), borrow_spec)
    oirf = occbin_irf(borrow_spec, borrow_constraint, 1, 60; magnitude=-40.0)
    save("occbin_irf.html", plot_result(oirf))

    # -------------------------------------------------------------------
    # 33. OccBin solution path (borrowing constraint)
    # -------------------------------------------------------------------
    occ_shocks = zeros(60, 1)
    occ_shocks[1, 1] = -40.0
    occ_sol = occbin_solve(borrow_spec, borrow_constraint; shock_path=occ_shocks)
    save("occbin_solution.html", plot_result(occ_sol))

    # -------------------------------------------------------------------
    # 34. DiD Event Study (DIDResult)
    # -------------------------------------------------------------------
    begin
        # Synthetic staggered DiD panel: 30 units, 20 periods, 3 cohorts
        n_units_did = 30; n_times_did = 20
        did_rows = []
        for i in 1:n_units_did
            g_time = i <= 10 ? 0 : (i <= 20 ? 8 : 14)
            for t in 1:n_times_did
                treat_effect = (g_time > 0 && t >= g_time) ? 2.0 + 0.3 * (t - g_time) : 0.0
                push!(did_rows, (group=i, time=t,
                                 y=randn() + treat_effect,
                                 treat_timing=Float64(g_time),
                                 x=randn()))
            end
        end
        did_df = DataFrame(did_rows)
        pd_did = xtset(did_df, :group, :time)
        did_result = estimate_did(pd_did, :y, :treat_timing;
                                  method=:callaway_santanna, leads=3, horizon=5)
        save("did_event_study.html", plot_result(did_result))
    end

    # -------------------------------------------------------------------
    # 35. Bacon Decomposition (BaconDecomposition)
    # -------------------------------------------------------------------
    begin
        bd = bacon_decomposition(pd_did, :y, :treat_timing)
        save("did_bacon.html", plot_result(bd))
    end

    # -------------------------------------------------------------------
    # 36. HonestDiD Sensitivity (HonestDiDResult)
    # -------------------------------------------------------------------
    begin
        hd = honest_did(did_result; Mbar=1.0)
        save("did_honest.html", plot_result(hd))
    end

    # -------------------------------------------------------------------
    # 37. Event Study LP (EventStudyLP)
    # -------------------------------------------------------------------
    begin
        eslp = estimate_event_study_lp(pd_did, :y, :treat_timing, 5; leads=3, lags=2)
        save("eslp_event_study.html", plot_result(eslp))
    end

    # -------------------------------------------------------------------
    # 38. OLS Regression Diagnostics
    # -------------------------------------------------------------------
    begin
        Random.seed!(123)
        n_reg = 200
        X_reg = hcat(ones(n_reg), randn(n_reg), randn(n_reg))
        y_reg = X_reg * [1.0, 2.0, 0.5] + 0.5 * randn(n_reg)
        m_ols = estimate_reg(y_reg, X_reg; varnames=["const", "x₁", "x₂"])
        save("reg_ols_diagnostics.html", plot_result(m_ols))
    end

    # -------------------------------------------------------------------
    # 39. IV/2SLS Regression
    # -------------------------------------------------------------------
    begin
        Random.seed!(124)
        n_iv = 300
        z1 = randn(n_iv); z2 = randn(n_iv)
        u = randn(n_iv)
        x_endog = 0.5 .* z1 .+ 0.3 .* z2 .+ 0.7 .* u
        x_exog = randn(n_iv)
        y_iv = 1.0 .+ 1.5 .* x_endog .+ 0.8 .* x_exog .+ u
        X_iv = hcat(ones(n_iv), x_endog, x_exog)
        Z_iv = hcat(ones(n_iv), z1, z2, x_exog)
        m_iv = estimate_iv(y_iv, X_iv, Z_iv;
                           endogenous=[2],
                           varnames=["const", "x_endog", "x_exog"])
        save("reg_iv.html", plot_result(m_iv))
    end

    # -------------------------------------------------------------------
    # 40. Logit Model
    # -------------------------------------------------------------------
    begin
        Random.seed!(125)
        n_bin = 500
        X_bin = hcat(ones(n_bin), randn(n_bin), randn(n_bin))
        eta = X_bin * [0.0, 1.0, -0.5]
        prob_bin = 1.0 ./ (1.0 .+ exp.(-eta))
        y_bin = Float64.(rand(n_bin) .< prob_bin)
        m_logit = estimate_logit(y_bin, X_bin; varnames=["const", "x₁", "x₂"])
        save("reg_logit.html", plot_result(m_logit))
    end

    # -------------------------------------------------------------------
    # 41. Probit Model
    # -------------------------------------------------------------------
    begin
        m_probit_reg = estimate_probit(y_bin, X_bin; varnames=["const", "x₁", "x₂"])
        save("reg_probit.html", plot_result(m_probit_reg))
    end

    # -------------------------------------------------------------------
    # 42. Marginal Effects
    # -------------------------------------------------------------------
    begin
        me_ame = marginal_effects(m_logit; type=:ame)
        save("reg_marginal_effects.html", plot_result(me_ame))
    end

    # -------------------------------------------------------------------
    # 43. Classification Diagnostics
    # -------------------------------------------------------------------
    begin
        save("reg_classification.html", plot_result(m_logit; title="Logit Classification Diagnostics"))
    end

    # -------------------------------------------------------------------
    # 44. ACF/PACF Correlogram (ACFResult)
    # -------------------------------------------------------------------
    begin
        y_acf = use_real ? filter(isfinite, diff(log.(fred_gm[:, "INDPRO"]))) : randn(200)
        r_acf = acf_pacf(y_acf; lags=24)
        save("spectral_acf.html", plot_result(r_acf))
    end

    # -------------------------------------------------------------------
    # 45. Spectral Density (SpectralDensityResult)
    # -------------------------------------------------------------------
    begin
        y_sd = use_real ? filter(isfinite, diff(log.(fred_gm[:, "INDPRO"]))) : randn(200)
        r_sd = spectral_density(y_sd; method=:welch)
        save("spectral_density.html", plot_result(r_sd))
    end

    # -------------------------------------------------------------------
    # 46. Cross-Spectrum Coherence + Phase (CrossSpectrumResult)
    # -------------------------------------------------------------------
    begin
        if use_real
            y_cs1 = filter(isfinite, diff(log.(fred_gm[:, "INDPRO"])))
            y_cs2 = filter(isfinite, diff(log.(fred_gm[:, "CPIAUCSL"])))
            n_cs = min(length(y_cs1), length(y_cs2))
            r_cs = cross_spectrum(y_cs1[1:n_cs], y_cs2[1:n_cs])
        else
            r_cs = cross_spectrum(randn(200), randn(200))
        end
        save("spectral_cross.html", plot_result(r_cs))
    end

    # -------------------------------------------------------------------
    # 47. Transfer Function — HP filter (TransferFunctionResult)
    # -------------------------------------------------------------------
    begin
        r_tf = transfer_function(:hp; lambda=1600)
        save("spectral_transfer.html", plot_result(r_tf))
    end

    # -------------------------------------------------------------------
    # 48. DSGE GIRF (2nd-order perturbation)
    # -------------------------------------------------------------------
    begin
        psol2 = perturbation_solver(dsge_sol.spec; order=2)
        girf = irf(psol2, 40; irf_type=:girf, n_draws=100)
        save("dsge_girf.html", plot_result(girf))
    end

    # -------------------------------------------------------------------
    # 49. Non-Gaussian SVAR IRF (ICA identification)
    # -------------------------------------------------------------------
    begin
        ng_irf = irf(m_var, 20; method=:fastica)
        save("nongaussian_irf.html", plot_result(ng_irf))
    end

    # -------------------------------------------------------------------
    # 50. Structural DFM IRF
    # -------------------------------------------------------------------
    begin
        sdfm = estimate_structural_dfm(X20, 3; p=1, H=20)
        r_sdfm = irf(sdfm, 20)
        save("sdfm_irf.html", plot_result(r_sdfm))
    end

    # -------------------------------------------------------------------
    # 51. Structural DFM Panel IRF
    # -------------------------------------------------------------------
    begin
        panel_irf = sdfm_panel_irf(sdfm, 20)
        save("sdfm_panel_irf.html", plot_result(panel_irf))
    end

    # -------------------------------------------------------------------
    # 52. Bayesian DSGE IRF (estimate_dsge_bayes + credible bands)
    # -------------------------------------------------------------------
    begin
        # Simulate data from the RBC model for Bayesian estimation
        Y_dsge_sim = simulate(dsge_sol, 200)
        Y_data = Y_dsge_sim[:, [1, 4]]  # observe Y and A

        bayes_result = estimate_dsge_bayes(dsge_spec, Y_data, [0.9];
            priors=Dict(:ρ => Beta(5, 2)),
            method=:mh, n_draws=5000, burnin=2000,
            observables=[:Y, :A], measurement_error=:auto)
        birf_dsge = irf(bayes_result, 40; n_draws=50)
        save("dsge_bayes_irf.html", plot_result(birf_dsge))
    end

    # -------------------------------------------------------------------
    # 53. DSGE Historical Decomposition (linear)
    # -------------------------------------------------------------------
    begin
        Y_dsge_hd = simulate(dsge_sol, 200)
        # 4 observables vs 1 structural shock: measurement error keeps the
        # observation covariance nonsingular (post-#139 stochastic-singularity guard)
        dsge_hd = historical_decomposition(dsge_sol, Y_dsge_hd, [:Y, :C, :K, :A];
                                           measurement_error=fill(0.01, 4))
        save("dsge_hd.html", plot_result(dsge_hd))
    end

    # ===================================================================
    # PLT-40 gallery — full dispatch set. Each block is guarded (try_save
    # or a topic-level try) so one estimator failure never aborts the run.
    # ===================================================================
    println("\n-- PLT-40 gallery assets --")

    # 54. Panel VAR — orthogonalized IRF + companion-eigenvalue stability
    try
        rng = MersenneTwister(7)
        N, Tt, mm = 18, 22, 2
        A1 = [0.35 0.05; -0.04 0.30] .+ 0.02 .* randn(rng, mm, mm)
        dm = zeros(N * Tt, mm)
        for i in 1:N
            mu = randn(rng, mm); off = (i - 1) * Tt; dm[off + 1, :] = mu
            for t in 2:Tt
                dm[off + t, :] = mu .+ A1 * dm[off + t - 1, :] .+ 0.10 .* randn(rng, mm)
            end
        end
        df_pv = DataFrame(dm, ["output", "prices"])
        df_pv.id = repeat(1:N, inner=Tt); df_pv.time = repeat(1:Tt, outer=N)
        pv = estimate_pvar_feols(xtset(df_pv, :id, :time), 1)
        save("pvar_irf.html", plot_result(pv; view=:oirf, H=12))
        save("pvar_stability.html", plot_result(pv; view=:stability))
    catch e
        @warn "Skipped PVAR gallery" exception=(e, catch_backtrace())
    end

    # 55. Set-identified SVAR — median response + nested identified-set band
    try_save("svar_setid_band.html") do
        H = 20; n = 2; nd = 300
        draws = randn(MersenneTwister(3), nd, H, n, n) .* 0.35
        for a in 1:nd, hh in 1:H
            draws[a, hh, :, :] .+= 0.9 * exp(-0.14 * (hh - 1))
        end
        sis = MEM.SignIdentifiedSet{Float64}([randn(n, n) for _ in 1:nd], draws, nd,
                600, nd / 600, ["output", "inflation"], ["demand", "supply"])
        plot_result(sis)
    end

    # 56. Markov-switching SVAR — smoothed regime probabilities (stacked area)
    try_save("ms_regime_probs.html") do
        Tt = 140; K = 2
        rp = zeros(Tt, K)
        for t in 1:Tt
            p1 = clamp(0.5 + 0.45 * sin(2π * t / 70), 0.02, 0.98)
            rp[t, 1] = p1; rp[t, 2] = 1 - p1
        end
        ms = MEM.MarkovSwitchingSVARResult([1.0 0.2; 0.3 1.0], Matrix(1.0I, 2, 2),
                [Matrix(1.0I, 2, 2), Matrix(2.0I, 2, 2)], [[1.0, 1.0], [1.5, 2.0]],
                rp, [0.9 0.1; 0.2 0.8], -100.0, true, 25, K)
        plot_result(ms; view=:regimes)
    end

    # 57. HA-DSGE (Krusell-Smith) — distribution IRF + inequality response
    try
        ha_spec = load_ha_example(:krusell_smith)
        ha_ss   = compute_steady_state(ha_spec; r_bounds=(-0.02, 0.04), max_iter=60, tol=1e-3)
        ha_sol  = solve(ha_spec; method=:reiter, ss=ha_ss, n_reduced=12)
        save("ha_distribution_dynamics.html", plot_result(ha_sol; horizon=16, max_bins=50))
        save("ha_inequality.html", plot_result(ha_sol; view=:inequality, horizon=16))
    catch e
        @warn "Skipped HA dynamics gallery" exception=(e, catch_backtrace())
    end

    # 58. Continuous-time Aiyagari — wealth distribution + policy functions
    try
        Ig = 60
        ag = collect(range(0.0, 15.0; length=Ig))
        gg = hcat(exp.(-0.30 .* ag), 0.6 .* exp.(-0.24 .* ag))
        da = ag[2] - ag[1]; gg ./= (sum(gg) * da)
        cg = hcat(0.5 .+ 0.30 .* ag, 0.7 .+ 0.30 .* ag)
        sg = hcat(0.10 .* (8 .- ag), 0.10 .* (9 .- ag))
        vg = -exp.(-0.10 .* hcat(ag, ag))
        ct_ss = MEM.CTSteadyState{Float64}(0.03, 1.2, 4.0, 1.0, ag, gg, vg, cg, sg,
                    spzeros(2Ig, 2Ig), true)
        save("ct_distribution.html", plot_result(ct_ss; view=:distribution))
        save("ct_policy.html", plot_result(ct_ss; view=:policy))
    catch e
        @warn "Skipped CT gallery" exception=(e, catch_backtrace())
    end

    # 59. Panel-regression coefficient forest (fixed effects)
    try_save("micro_coef_forest.html") do
        rng = MersenneTwister(61); Np, Tp = 20, 15
        dfm = DataFrame(id=repeat(1:Np, inner=Tp), time=repeat(1:Tp, outer=Np))
        dfm.capital = randn(rng, Np * Tp); dfm.labor = randn(rng, Np * Tp)
        dfm.rnd = randn(rng, Np * Tp)
        dfm.output = 0.5 .* dfm.capital .- 0.3 .* dfm.labor .+ 0.2 .* dfm.rnd .+
                     randn(rng, Np * Tp)
        preg = estimate_xtreg(xtset(dfm, :id, :time), :output, [:capital, :labor, :rnd])
        plot_result(preg)
    end

    # 60. Odds-ratio forest plot (logit, log x-axis, reference at 1)
    try_save("odds_ratio_forest.html") do
        rng = MersenneTwister(62); nb = 500
        Xb = hcat(randn(rng, nb), randn(rng, nb), randn(rng, nb))
        eta = Xb * [0.8, -0.5, 0.3]
        yb = Float64.(rand(rng, nb) .< 1 ./ (1 .+ exp.(-eta)))
        plot_result(odds_ratio(estimate_logit(yb, Xb; varnames=["age", "income", "educ"])))
    end

    # 61. GMM moment-discrepancy bar + J-test annotation
    try_save("gmm_moment_fit.html") do
        rng = MersenneTwister(63); ng = 300
        Xg = randn(rng, ng, 2); yg = Xg * [1.0, -0.5] .+ randn(rng, ng)
        Zg = hcat(Xg, randn(rng, ng)); datag = hcat(yg, Zg)
        mfn = (theta, d) -> d[:, 2:4] .* (d[:, 1] .- d[:, 2:3] * theta)
        plot_result(estimate_gmm(mfn, [0.0, 0.0], datag; weighting=:two_step))
    end

    # 62. GARCH news-impact curve (view=:news_impact on the existing GARCH fit)
    try_save("news_impact_curve.html") do
        plot_result(gm; view=:news_impact)
    end

    # 63. Mincer-Zarnowitz forecast-efficiency line vs the 45° reference
    try_save("mincer_zarnowitz.html") do
        rng = MersenneTwister(64); a_mz = randn(rng, 80)
        f_mz = 0.2 .+ 0.9 .* a_mz .+ 0.3 .* randn(rng, 80)
        plot_result(mincer_zarnowitz(a_mz, f_mz))
    end

    # 64. State-dependent local projection — expansion vs recession IRFs
    try_save("state_lp.html") do
        Yl = randn(MersenneTwister(51), 160, 2)
        statev = cumsum(randn(MersenneTwister(52), 160))
        slp = estimate_state_lp(Yl, 1, statev, 12; lags=3, varnames=["activity", "spread"])
        plot_result(slp)
    end

    # 65. Leontief inverse heatmap (sequential single-hue ramp + color legend)
    try_save("leontief_heatmap.html") do
        plot_result(leontief(load_example(:wiot)))
    end

    # 66. TimeSeriesData correlation heatmap (view=:corr)
    try_save("data_timeseries_corr.html") do
        if use_real
            cv = [v for v in ["INDPRO", "CPIAUCSL", "FEDFUNDS", "UNRATE", "M2SL",
                              "GS10", "PAYEMS", "TB3MS"] if v in varnames(fred_gm)]
            Xc = to_matrix(apply_tcode(fred_gm[:, cv]))
            gc = [j for j in 1:size(Xc, 2) if !any(isnan, Xc[:, j])]
            TimeSeriesData(_clean_rows(Xc[:, gc]); varnames=cv[gc])
        else
            d_ts
        end |> d -> plot_result(d; view=:corr)
    end

    # 67. Binscatter with controls (CrossSectionData view=:binscatter)
    try_save("binscatter.html") do
        rng = MersenneTwister(65); nbc = 800
        xb = randn(rng, nbc); zb = randn(rng, nbc)
        yb = 0.7 .* xb .+ 0.4 .* zb .+ randn(rng, nbc)
        csb = CrossSectionData(hcat(xb, yb, zb); varnames=["schooling", "wage", "ability"])
        plot_result(csb; view=:binscatter, x="schooling", y="wage", controls=["ability"])
    end

    # 68. MCMC trace (BVAR posterior draws, view=:trace)
    try_save("mcmc_trace.html") do
        plot_result(post; view=:trace, params=[1, 2, 3])
    end

    # 69. Prior/posterior overlap diagnostic (identification screen)
    try_save("prior_posterior.html") do
        o = MEM.PriorPosteriorOverlap{Float64}([:β, :α, :δ, :ρ, :σ],
                [0.28, 0.62, 0.91, 0.45, 0.83], [false, false, true, false, true], 0.8)
        plot_result(o)
    end

    println("\nDone! Generated $(length(readdir(PLOT_DIR))) HTML files in $PLOT_DIR")
end

main()
