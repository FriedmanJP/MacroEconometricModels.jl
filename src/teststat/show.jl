# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Publication-quality show methods for unit root test results using PrettyTables.
"""

# _significance_stars, _format_pvalue are defined in display_utils.jl

function Base.show(io::IO, r::ADFResult)
    spec_data = Any[
        "H₀" "Series has a unit root (non-stationary)";
        "H₁" "Series is stationary";
        "Deterministic terms" _regression_name(r.regression);
        "Lag length" r.lags;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Augmented Dickey-Fuller Unit Root Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "Test statistic (τ)" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
    cv_data = Matrix{Any}(undef, 1, 3)
    cv_data[1, :] = [round(r.critical_values[1], digits=3),
                     round(r.critical_values[5], digits=3),
                     round(r.critical_values[10], digits=3)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
    )
    reject_1 = r.statistic < r.critical_values[1]
    reject_5 = r.statistic < r.critical_values[5]
    reject_10 = r.statistic < r.critical_values[10]
    conclusion = if reject_1
        "Reject H₀ at 1% significance level"
    elseif reject_5
        "Reject H₀ at 5% significance level"
    elseif reject_10
        "Reject H₀ at 10% significance level"
    else
        "Fail to reject H₀ (series appears non-stationary)"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::KPSSResult)
    stationarity_type = r.regression == :constant ? "level" : "trend"
    spec_data = Any[
        "H₀" string("Series is ", stationarity_type, " stationary");
        "H₁" "Series has a unit root";
        "Deterministic terms" _regression_name(r.regression);
        "Bandwidth (Bartlett)" r.bandwidth;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "KPSS Stationarity Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    pval_display = r.pvalue < 0.01 ? "<0.01" : (r.pvalue > 0.10 ? ">0.10" : string(round(r.pvalue, digits=4)))
    results_data = Any[
        "LM statistic" string(round(r.statistic, digits=4), " ", stars);
        "P-value" pval_display
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
    cv_data = Matrix{Any}(undef, 1, 3)
    # Critical-value order standardized to 1% / 5% / 10% package-wide (S8/T166).
    cv_data[1, :] = [_fmt(r.critical_values[1]; digits=3),
                     _fmt(r.critical_values[5]; digits=3),
                     _fmt(r.critical_values[10]; digits=3)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
    )
    reject_1 = r.statistic > r.critical_values[1]
    reject_5 = r.statistic > r.critical_values[5]
    reject_10 = r.statistic > r.critical_values[10]
    conclusion = if reject_1
        "Reject H₀ at 1% level (series is non-stationary)"
    elseif reject_5
        "Reject H₀ at 5% level (series is non-stationary)"
    elseif reject_10
        "Reject H₀ at 10% level (series is non-stationary)"
    else
        "Fail to reject H₀ (series appears stationary)"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::PPResult)
    spec_data = Any[
        "H₀" "Series has a unit root (non-stationary)";
        "H₁" "Series is stationary";
        "Deterministic terms" _regression_name(r.regression);
        "Bandwidth (Newey-West)" r.bandwidth;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Phillips-Perron Unit Root Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "Adj. t-statistic (Zₜ)" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
    cv_data = Matrix{Any}(undef, 1, 3)
    cv_data[1, :] = [round(r.critical_values[1], digits=3),
                     round(r.critical_values[5], digits=3),
                     round(r.critical_values[10], digits=3)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
    )
    reject_1 = r.statistic < r.critical_values[1]
    reject_5 = r.statistic < r.critical_values[5]
    reject_10 = r.statistic < r.critical_values[10]
    conclusion = if reject_1
        "Reject H₀ at 1% significance level"
    elseif reject_5
        "Reject H₀ at 5% significance level"
    elseif reject_10
        "Reject H₀ at 10% significance level"
    else
        "Fail to reject H₀ (series appears non-stationary)"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l], )
end

function Base.show(io::IO, r::ZAResult)
    break_type = r.regression == :constant ? "intercept" : (r.regression == :trend ? "trend" : "intercept and trend")
    spec_data = Any[
        "H₀" "Series has a unit root without structural break";
        "H₁" string("Series is stationary with break in ", break_type);
        "Break type" _regression_name(r.regression);
        "Lag length" r.lags;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Zivot-Andrews Unit Root Test with Structural Break",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    break_pct = string(round(r.break_fraction * 100, digits=1), "% of sample")
    break_data = Any[
        "Break index" r.break_index;
        "Break location" break_pct
    ]
    _pretty_table(io, break_data;
        title = "Estimated Break Point",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    pval_display = r.pvalue < 0.01 ? "<0.01" : string(round(r.pvalue, digits=4))
    results_data = Any[
        "Minimum t-statistic" string(round(r.statistic, digits=4), " ", stars);
        "P-value" pval_display
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
    cv_data = Matrix{Any}(undef, 1, 3)
    cv_data[1, :] = [round(r.critical_values[1], digits=2),
                     round(r.critical_values[5], digits=2),
                     round(r.critical_values[10], digits=2)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
    )
    reject_1 = r.statistic < r.critical_values[1]
    reject_5 = r.statistic < r.critical_values[5]
    reject_10 = r.statistic < r.critical_values[10]
    conclusion = if reject_1
        "Reject H₀ at 1% level (stationary with break)"
    elseif reject_5
        "Reject H₀ at 5% level (stationary with break)"
    elseif reject_10
        "Reject H₀ at 10% level (stationary with break)"
    else
        "Fail to reject H₀ (unit root, no significant break)"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::NgPerronResult)
    spec_data = Any[
        "H₀" "Series has a unit root (non-stationary)";
        "H₁" "Series is stationary";
        "Deterministic terms" _regression_name(r.regression);
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Ng-Perron Unit Root Tests (GLS Detrended)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    mza_reject_5 = r.MZa < r.critical_values[:MZa][5]
    mza_reject_1 = r.MZa < r.critical_values[:MZa][1]
    mza_reject_10 = r.MZa < r.critical_values[:MZa][10]
    mza_stars = mza_reject_1 ? "***" : (mza_reject_5 ? "**" : (mza_reject_10 ? "*" : ""))
    mzt_reject_5 = r.MZt < r.critical_values[:MZt][5]
    mzt_reject_1 = r.MZt < r.critical_values[:MZt][1]
    mzt_reject_10 = r.MZt < r.critical_values[:MZt][10]
    mzt_stars = mzt_reject_1 ? "***" : (mzt_reject_5 ? "**" : (mzt_reject_10 ? "*" : ""))
    msb_reject_5 = r.MSB < r.critical_values[:MSB][5]
    msb_reject_1 = r.MSB < r.critical_values[:MSB][1]
    msb_reject_10 = r.MSB < r.critical_values[:MSB][10]
    msb_stars = msb_reject_1 ? "***" : (msb_reject_5 ? "**" : (msb_reject_10 ? "*" : ""))
    mpt_reject_5 = r.MPT < r.critical_values[:MPT][5]
    mpt_reject_1 = r.MPT < r.critical_values[:MPT][1]
    mpt_reject_10 = r.MPT < r.critical_values[:MPT][10]
    mpt_stars = mpt_reject_1 ? "***" : (mpt_reject_5 ? "**" : (mpt_reject_10 ? "*" : ""))
    stats_data = Any[
        "MZα" string(round(r.MZa, digits=4), " ", mza_stars) round(r.critical_values[:MZa][1], digits=2) round(r.critical_values[:MZa][5], digits=2) round(r.critical_values[:MZa][10], digits=2);
        "MZₜ" string(round(r.MZt, digits=4), " ", mzt_stars) round(r.critical_values[:MZt][1], digits=2) round(r.critical_values[:MZt][5], digits=2) round(r.critical_values[:MZt][10], digits=2);
        "MSB" string(round(r.MSB, digits=4), " ", msb_stars) round(r.critical_values[:MSB][1], digits=3) round(r.critical_values[:MSB][5], digits=3) round(r.critical_values[:MSB][10], digits=3);
        "MPT" string(round(r.MPT, digits=4), " ", mpt_stars) round(r.critical_values[:MPT][1], digits=2) round(r.critical_values[:MPT][5], digits=2) round(r.critical_values[:MPT][10], digits=2)
    ]
    _pretty_table(io, stats_data;
        title = "Test Statistics",
        column_labels = ["Statistic", "Value", "1% CV", "5% CV", "10% CV"],
        alignment = [:l, :r, :r, :r, :r],
    )
    n_reject_5 = sum([mza_reject_5, mzt_reject_5, msb_reject_5, mpt_reject_5])
    conclusion = if n_reject_5 >= 3
        "Strong evidence against unit root (reject H₀)"
    elseif n_reject_5 >= 2
        "Moderate evidence against unit root"
    elseif n_reject_5 >= 1
        "Weak evidence against unit root"
    else
        "Fail to reject H₀ (series appears non-stationary)"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::JohansenResult)
    n = length(r.trace_stats)
    det_name = r.deterministic == :none ? "No deterministic terms" :
               r.deterministic == :constant ? "Constant in cointegrating equation" :
               "Linear trend in data"
    spec_data = Any[
        "Deterministic terms" det_name;
        "Lags in VECM" r.lags;
        "Observations" r.nobs;
        "Number of variables" n
    ]
    _pretty_table(io, spec_data;
        title = "Johansen Cointegration Test",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    trace_data = Matrix{Any}(undef, n, 5)
    for i in 1:n
        rank = i - 1
        stat = r.trace_stats[i]
        cv = r.critical_values_trace[i, 2]
        pval = r.trace_pvalues[i]
        reject_5 = stat > cv
        reject_1 = stat > r.critical_values_trace[i, 3]
        reject_10 = stat > r.critical_values_trace[i, 1]
        stars = reject_1 ? "***" : (reject_5 ? "**" : (reject_10 ? "*" : ""))
        pval_str = _format_pvalue(pval)
        trace_data[i, 1] = rank
        trace_data[i, 2] = string(round(stat, digits=2), " ", stars)
        trace_data[i, 3] = round(cv, digits=2)
        trace_data[i, 4] = pval_str
        trace_data[i, 5] = reject_5 ? "Reject" : ""
    end
    _pretty_table(io, trace_data;
        title = "Trace Test",
        column_labels = ["H₀: rank ≤ r", "Statistic", "5% CV", "P-value", "Decision"],
        alignment = [:r, :r, :r, :r, :l],
    )
    max_data = Matrix{Any}(undef, n, 5)
    for i in 1:n
        rank = i - 1
        stat = r.max_eigen_stats[i]
        cv = r.critical_values_max[i, 2]
        pval = r.max_eigen_pvalues[i]
        reject_5 = stat > cv
        reject_1 = stat > r.critical_values_max[i, 3]
        reject_10 = stat > r.critical_values_max[i, 1]
        stars = reject_1 ? "***" : (reject_5 ? "**" : (reject_10 ? "*" : ""))
        pval_str = _format_pvalue(pval)
        max_data[i, 1] = rank
        max_data[i, 2] = string(round(stat, digits=2), " ", stars)
        max_data[i, 3] = round(cv, digits=2)
        max_data[i, 4] = pval_str
        max_data[i, 5] = reject_5 ? "Reject" : ""
    end
    _pretty_table(io, max_data;
        title = "Maximum Eigenvalue Test",
        column_labels = ["H₀: rank = r", "Statistic", "5% CV", "P-value", "Decision"],
        alignment = [:r, :r, :r, :r, :l],
    )
    eig_data = Matrix{Any}(undef, 1, n)
    for i in 1:n
        eig_data[1, i] = round(r.eigenvalues[i], digits=4)
    end
    _pretty_table(io, eig_data;
        title = "Eigenvalues",
        column_labels = ["λ$i" for i in 1:n],
        alignment = :r,
        row_labels = [""]
    )
    conclusion = if r.rank == 0
        "No cointegrating relationships found"
    elseif r.rank == n
        "All variables are stationary (full rank)"
    else
        string("Estimated cointegration rank = ", r.rank)
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::VARStationarityResult)
    n_eigs = length(r.eigenvalues)
    n_show = min(n_eigs, 10)
    moduli = abs.(r.eigenvalues)
    sorted_idx = sortperm(moduli, rev=true)
    nrows = n_eigs > 10 ? n_show + 1 : n_show
    eig_data = Matrix{Any}(undef, nrows, 3)
    for i in 1:n_show
        idx = sorted_idx[i]
        λ = r.eigenvalues[idx]
        mod = moduli[idx]
        eig_data[i, 1] = i
        if imag(λ) ≈ 0
            eig_data[i, 2] = round(real(λ), digits=4)
        else
            sign_str = imag(λ) >= 0 ? "+" : "-"
            eig_data[i, 2] = string(round(real(λ), digits=4), sign_str, round(abs(imag(λ)), digits=4), "i")
        end
        eig_data[i, 3] = round(mod, digits=4)
    end
    if n_eigs > 10
        eig_data[nrows, 1] = "..."
        eig_data[nrows, 2] = string("(", n_eigs - 10, " more)")
        eig_data[nrows, 3] = ""
    end
    _pretty_table(io, eig_data;
        title = "VAR Model Stationarity Test — Companion Matrix Eigenvalues",
        column_labels = ["Index", "Eigenvalue", "Modulus"],
        alignment = [:r, :r, :r],
    )
    result_str = r.is_stationary ?
        "VAR is STATIONARY (all eigenvalue moduli < 1)" :
        "VAR is NON-STATIONARY (maximum eigenvalue modulus ≥ 1)"
    summary_data = Any[
        "Maximum modulus" round(r.max_modulus, digits=6);
        "Number of eigenvalues" n_eigs;
        "Stationary" (r.is_stationary ? "Yes" : "No");
        "Result" result_str
    ]
    _pretty_table(io, summary_data;
        title = "Summary",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::FourierADFResult)
    spec_data = Any[
        "H₀" "Series has a unit root";
        "H₁" "Series is stationary (with smooth breaks)";
        "Deterministic terms" _regression_name(r.regression);
        "Fourier frequency (k)" r.frequency;
        "Lag length" r.lags;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Fourier ADF Unit Root Test (Enders & Lee 2012)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    f_stars = _significance_stars(r.f_pvalue)
    results_data = Any[
        "ADF statistic (τ)" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue);
        "F-statistic (Fourier terms)" string(round(r.f_statistic, digits=4), " ", f_stars);
        "F p-value" _format_pvalue(r.f_pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
    cv_data = Matrix{Any}(undef, 2, 3)
    cv_data[1, :] = [round(r.critical_values[1], digits=3),
                     round(r.critical_values[5], digits=3),
                     round(r.critical_values[10], digits=3)]
    cv_data[2, :] = [round(r.f_critical_values[1], digits=3),
                     round(r.f_critical_values[5], digits=3),
                     round(r.f_critical_values[10], digits=3)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
        row_labels = ["ADF τ", "F-test"]
    )
    reject_5 = r.statistic < r.critical_values[5]
    conclusion = reject_5 ? "Reject H₀ at 5% level (stationary with smooth breaks)" :
                            "Fail to reject H₀ (unit root)"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::FourierKPSSResult)
    stationarity_type = r.regression == :constant ? "level" : "trend"
    spec_data = Any[
        "H₀" string("Series is ", stationarity_type, " stationary (with smooth breaks)");
        "H₁" "Series has a unit root";
        "Deterministic terms" _regression_name(r.regression);
        "Fourier frequency (k)" r.frequency;
        "Bandwidth (Bartlett)" r.bandwidth;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Fourier KPSS Stationarity Test (Becker, Enders & Lee 2006)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "KPSS statistic" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue);
        "F-statistic (Fourier terms)" string(round(r.f_statistic, digits=4));
        "F p-value" _format_pvalue(r.f_pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
    cv_data = Matrix{Any}(undef, 1, 3)
    cv_data[1, :] = [round(r.critical_values[1], digits=4),
                     round(r.critical_values[5], digits=4),
                     round(r.critical_values[10], digits=4)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
    )
    reject_5 = r.statistic > r.critical_values[5]
    conclusion = reject_5 ? "Reject H₀ at 5% level (unit root)" :
                            "Fail to reject H₀ (series appears stationary)"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::DFGLSResult)
    spec_data = Any[
        "H₀" "Series has a unit root";
        "H₁" "Series is stationary";
        "Deterministic terms" _regression_name(r.regression);
        "Lag length" r.lags;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "DF-GLS Unit Root Test (Elliott, Rothenberg & Stock 1996)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    pt_stars = _significance_stars(r.pt_pvalue)
    results_data = Any[
        "DF-GLS τ statistic" string(round(r.statistic, digits=4), " ", stars);
        "DF-GLS p-value" _format_pvalue(r.pvalue);
        "ERS Pt statistic" string(round(r.pt_statistic, digits=4), " ", pt_stars);
        "Pt p-value" _format_pvalue(r.pt_pvalue);
        "MZα" string(round(r.MZa, digits=4));
        "MZt" string(round(r.MZt, digits=4));
        "MSB" string(round(r.MSB, digits=4));
        "MPT" string(round(r.MPT, digits=4))
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
    cv_data = Matrix{Any}(undef, 2, 3)
    cv_data[1, :] = [round(r.critical_values[1], digits=3),
                     round(r.critical_values[5], digits=3),
                     round(r.critical_values[10], digits=3)]
    cv_data[2, :] = [round(r.pt_critical_values[1], digits=3),
                     round(r.pt_critical_values[5], digits=3),
                     round(r.pt_critical_values[10], digits=3)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
        row_labels = ["DF-GLS τ", "ERS Pt"]
    )
    reject_5 = r.statistic < r.critical_values[5]
    conclusion = reject_5 ? "Reject H₀ at 5% level (stationary)" :
                            "Fail to reject H₀ (unit root)"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::LMUnitRootResult)
    break_desc = r.breaks == 0 ? "None" : string(r.breaks)
    reg_desc = r.regression == :level ? "Level shift" : "Level + trend shift"
    spec_data = Any[
        "H₀" "Series has a unit root (breaks under H₀)";
        "H₁" "Series is stationary";
        "Structural breaks" break_desc;
        "Break type" reg_desc;
        "Lag length" r.lags;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "LM Unit Root Test (Lee & Strazicich)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    if r.breaks > 0
        break_data = Matrix{Any}(undef, r.breaks, 2)
        for i in 1:r.breaks
            break_data[i, 1] = r.break_dates[i]
            break_data[i, 2] = string(round(r.break_fractions[i] * 100, digits=1), "% of sample")
        end
        _pretty_table(io, break_data;
            title = "Estimated Break Points",
            column_labels = ["Index", "Location"],
            alignment = [:r, :r],
        )
    end
    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "LM statistic (τ)" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
    cv_data = Matrix{Any}(undef, 1, 3)
    cv_data[1, :] = [round(r.critical_values[1], digits=3),
                     round(r.critical_values[5], digits=3),
                     round(r.critical_values[10], digits=3)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
    )
    reject_5 = r.statistic < r.critical_values[5]
    conclusion = reject_5 ? "Reject H₀ at 5% level (stationary)" :
                            "Fail to reject H₀ (unit root)"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::ADF2BreakResult)
    model_desc = r.model == :level ? "level shifts" : "level + trend shifts"
    spec_data = Any[
        "H₀" "Series has a unit root with two breaks";
        "H₁" string("Series is stationary with two breaks (", model_desc, ")");
        "Model" (r.model == :level ? "A (level shifts)" : "C (level + trend shifts)");
        "Lag length" r.lags;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Two-Break ADF Unit Root Test (Narayan & Popp 2010)",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    break1_pct = string(round(r.break1_fraction * 100, digits=1), "% of sample")
    break2_pct = string(round(r.break2_fraction * 100, digits=1), "% of sample")
    break_data = Any[
        "Break 1 index" r.break1;
        "Break 1 location" break1_pct;
        "Break 2 index" r.break2;
        "Break 2 location" break2_pct
    ]
    _pretty_table(io, break_data;
        title = "Estimated Break Points",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "Minimum t-statistic" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results",
        column_labels = ["", "Value"],
        alignment = [:l, :r],
    )
    cv_data = Matrix{Any}(undef, 1, 3)
    cv_data[1, :] = [round(r.critical_values[1], digits=3),
                     round(r.critical_values[5], digits=3),
                     round(r.critical_values[10], digits=3)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
    )
    reject_1 = r.statistic < r.critical_values[1]
    reject_5 = r.statistic < r.critical_values[5]
    reject_10 = r.statistic < r.critical_values[10]
    conclusion = if reject_1
        "Reject H₀ at 1% level (stationary with two breaks)"
    elseif reject_5
        "Reject H₀ at 5% level (stationary with two breaks)"
    elseif reject_10
        "Reject H₀ at 10% level (stationary with two breaks)"
    else
        "Fail to reject H₀ (unit root, no significant breaks)"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::GregoryHansenResult)
    model_desc = r.model == :C ? "C (level shift)" :
                 r.model == :CT ? "C/T (level + trend shift)" : "C/S (regime shift)"
    spec_data = Any[
        "H₀" "No cointegration";
        "H₁" "Cointegration with a structural break";
        "Model" model_desc;
        "Regressors (m)" r.n_regressors;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Gregory-Hansen Cointegration Test with Structural Break",
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    adf_stars = _significance_stars(r.adf_pvalue)
    zt_stars = _significance_stars(r.zt_pvalue)
    za_stars = _significance_stars(r.za_pvalue)
    stats_data = Any[
        "ADF*" string(round(r.adf_statistic, digits=4), " ", adf_stars) r.adf_break _format_pvalue(r.adf_pvalue);
        "Zt*" string(round(r.zt_statistic, digits=4), " ", zt_stars) r.zt_break _format_pvalue(r.zt_pvalue);
        "Za*" string(round(r.za_statistic, digits=4), " ", za_stars) r.za_break _format_pvalue(r.za_pvalue)
    ]
    _pretty_table(io, stats_data;
        title = "Test Statistics",
        column_labels = ["Statistic", "Value", "Break", "P-value"],
        alignment = [:l, :r, :r, :r],
    )
    cv_data = Matrix{Any}(undef, 2, 3)
    cv_data[1, :] = [round(r.adf_critical_values[1], digits=3),
                     round(r.adf_critical_values[5], digits=3),
                     round(r.adf_critical_values[10], digits=3)]
    cv_data[2, :] = [round(r.za_critical_values[1], digits=2),
                     round(r.za_critical_values[5], digits=2),
                     round(r.za_critical_values[10], digits=2)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
        row_labels = ["ADF*/Zt*", "Za*"]
    )
    reject_adf = r.adf_statistic < r.adf_critical_values[5]
    reject_zt = r.zt_statistic < r.adf_critical_values[5]
    reject_za = r.za_statistic < r.za_critical_values[5]
    n_reject = sum([reject_adf, reject_zt, reject_za])
    conclusion = if n_reject >= 2
        "Reject H₀ (cointegration with structural break)"
    elseif n_reject == 1
        "Mixed evidence for cointegration with break"
    else
        "Fail to reject H₀ (no cointegration)"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# First-Generation Panel Unit Root Tests (EV-20, #428)
# LLC / IPS / Breitung / Fisher share the UNIT-ROOT null (left-tailed N(0,1));
# Hadri flips it (STATIONARITY null, right-tailed). Do not copy conclusions.
# =============================================================================

# Standard-normal critical values for a left-tailed test.
const _NORMAL_LEFT_CV = Dict(1 => -2.326, 5 => -1.645, 10 => -1.282)
# ... and right-tailed (Hadri).
const _NORMAL_RIGHT_CV = Dict(1 => 2.326, 5 => 1.645, 10 => 1.282)

function _panel_ur_cv_table(io::IO, cv::Dict)
    cv_data = Matrix{Any}(undef, 1, 3)
    cv_data[1, :] = [round(cv[1], digits=3), round(cv[5], digits=3), round(cv[10], digits=3)]
    _pretty_table(io, cv_data;
        title = "Critical Values (N(0,1))",
        column_labels = ["1%", "5%", "10%"],
        alignment = :r,
    )
end

function Base.show(io::IO, r::LLCResult{T}) where {T}
    spec_data = Any[
        "H0"          "All panels have a unit root (non-stationary)";
        "H1"          "All panels are stationary";
        "Deterministic" _regression_name(r.deterministic);
        "Avg. lags (p̄)" round(mean(r.lags), digits=2);
        "Adj. sample (T̃)" round(r.T_tilde, digits=2);
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Levin-Lin-Chu (2002) Panel Unit Root Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "Unadjusted t"       round(r.t_unadjusted, digits=4);
        "Adjusted t* (N(0,1))" string(round(r.statistic, digits=4), " ", stars);
        "P-value"            _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data; title = "Results",
        column_labels = ["", "Value"], alignment = [:l, :r])
    _panel_ur_cv_table(io, _NORMAL_LEFT_CV)
    reject = r.statistic < _NORMAL_LEFT_CV[5]
    conclusion = reject ?
        "Reject H0 at 5%: evidence panels are stationary" :
        "Fail to reject H0: panels appear non-stationary"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::IPSResult{T}) where {T}
    spec_data = Any[
        "H0"          "All panels have a unit root (non-stationary)";
        "H1"          "Some panels are stationary";
        "Deterministic" _regression_name(r.deterministic);
        "Avg. lags (p̄)" round(mean(r.lags), digits=2);
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Im-Pesaran-Shin (2003) Panel Unit Root Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "t-bar"                round(r.tbar, digits=4);
        "W_t-bar (N(0,1))"     string(round(r.statistic, digits=4), " ", stars);
        "P-value"              _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data; title = "Results",
        column_labels = ["", "Value"], alignment = [:l, :r])
    _panel_ur_cv_table(io, _NORMAL_LEFT_CV)
    reject = r.statistic < _NORMAL_LEFT_CV[5]
    conclusion = reject ?
        "Reject H0 at 5%: evidence some panels are stationary" :
        "Fail to reject H0: panels appear non-stationary"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::BreitungPanelResult{T}) where {T}
    spec_data = Any[
        "H0"          "All panels have a unit root (non-stationary)";
        "H1"          "All panels are stationary";
        "Deterministic" _regression_name(r.deterministic);
        "Prewhitening lags" r.lags;
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Breitung (2000) Panel Unit Root Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "lambda (N(0,1))" string(round(r.statistic, digits=4), " ", stars);
        "P-value"         _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data; title = "Results",
        column_labels = ["", "Value"], alignment = [:l, :r])
    _panel_ur_cv_table(io, _NORMAL_LEFT_CV)
    reject = r.statistic < _NORMAL_LEFT_CV[5]
    conclusion = reject ?
        "Reject H0 at 5%: evidence panels are stationary" :
        "Fail to reject H0: panels appear non-stationary"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::FisherPanelResult{T}) where {T}
    base_name = r.base == :adf ? "ADF (Dickey-Fuller)" : "Phillips-Perron"
    spec_data = Any[
        "H0"          "All panels have a unit root (non-stationary)";
        "H1"          "Some panels are stationary";
        "Per-unit test" base_name;
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Fisher-type (Maddala-Wu / Choi) Panel Unit Root Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    results_data = Any[
        "P (Maddala-Wu, χ²(2N))"  string(round(r.P, digits=4), " ", _significance_stars(r.P_pvalue))  _format_pvalue(r.P_pvalue);
        "Z (Choi inv-normal)"     string(round(r.Z, digits=4), " ", _significance_stars(r.Z_pvalue))  _format_pvalue(r.Z_pvalue);
        "L* (Choi logit, t)"      string(round(r.Lstar, digits=4), " ", _significance_stars(r.Lstar_pvalue))  _format_pvalue(r.Lstar_pvalue);
        "Pm (Choi modified)"      string(round(r.Pm, digits=4), " ", _significance_stars(r.Pm_pvalue))  _format_pvalue(r.Pm_pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Combination Statistics (primary: :$(r.combine))",
        column_labels = ["Statistic", "Value", "P-value"], alignment = [:l, :r, :r])
    reject = r.pvalue < 0.05
    conclusion = reject ?
        "Reject H0 at 5% ($(r.combine)): evidence some panels are stationary" :
        "Fail to reject H0 ($(r.combine)): panels appear non-stationary"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::HadriResult{T}) where {T}
    spec_data = Any[
        "H0"          "All panels are stationary";
        "H1"          "Some panels have a unit root (non-stationary)";
        "Deterministic" _regression_name(r.deterministic);
        "Variance"     (r.hetero ? "Heteroskedastic (per-unit σ̂²)" : "Homoskedastic (pooled σ̂²)");
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Hadri (2000) LM Panel Stationarity Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "LM-bar"          round(r.LM, digits=4);
        "Z (N(0,1))"      string(round(r.statistic, digits=4), " ", stars);
        "P-value"         _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data; title = "Results (right-tailed)",
        column_labels = ["", "Value"], alignment = [:l, :r])
    _panel_ur_cv_table(io, _NORMAL_RIGHT_CV)
    reject = r.statistic > _NORMAL_RIGHT_CV[5]
    conclusion = reject ?
        "Reject H0 at 5%: evidence some panels have a unit root" :
        "Fail to reject H0: panels appear stationary"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# Panel Cointegration Tests (EV-21, #429)
# H0: no cointegration (except: no direct H0 flip — all four share the null).
# Pedroni panel-v is RIGHT-tailed (large positive rejects); the other six
# Pedroni stats + Kao + Westerlund are LEFT-tailed.
# =============================================================================

function Base.show(io::IO, r::PedroniResult{T}) where {T}
    spec_data = Any[
        "H0"          "No cointegration in any panel";
        "H1"          "Cointegration (homogeneous / heterogeneous)";
        "Deterministic" _regression_name(r.trend);
        "Regressors (k)" r.n_regressors;
        "NW bandwidth" r.bandwidth;
        "ADF lags"     r.adf_lags;
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Pedroni (1999, 2004) Residual-Based Panel Cointegration Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    nstat = length(r.names)
    results_data = Matrix{Any}(undef, nstat, 4)
    for s in 1:nstat
        tail = s == 1 ? "right" : "left"
        results_data[s, 1] = r.names[s]
        results_data[s, 2] = round(r.raw[s], digits=4)
        results_data[s, 3] = string(round(r.statistics[s], digits=4), " ",
                                    _significance_stars(r.pvalues[s]))
        results_data[s, 4] = string(_format_pvalue(r.pvalues[s]), " (", tail, ")")
    end
    _pretty_table(io, results_data;
        title = "Statistics (standardized N(0,1); panel-v right-tailed)",
        column_labels = ["Statistic", "Raw", "Std (z)", "P-value"],
        alignment = [:l, :r, :r, :r])
    reject = r.pvalues[4] < 0.05      # panel-ADF is the workhorse
    conclusion = reject ?
        "Reject H0 at 5% (panel-ADF): evidence of cointegration" :
        "Fail to reject H0 (panel-ADF): no evidence of cointegration"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::KaoResult{T}) where {T}
    spec_data = Any[
        "H0"          "No cointegration (unit root in pooled residual)";
        "H1"          "Cointegration (homogeneous vector)";
        "Regressors (k)" r.n_regressors;
        "ADF lags (p)" r.lags;
        "Kernel lags"  r.kernel_lags;
        "ρ̂ (pooled)"   round(r.rho, digits=4);
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Kao (1999) Residual-Based Panel Cointegration Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    nstat = length(r.names)
    results_data = Matrix{Any}(undef, nstat, 3)
    for s in 1:nstat
        results_data[s, 1] = r.names[s]
        results_data[s, 2] = string(round(r.statistics[s], digits=4), " ",
                                    _significance_stars(r.pvalues[s]))
        results_data[s, 3] = _format_pvalue(r.pvalues[s])
    end
    _pretty_table(io, results_data;
        title = "Statistics (all N(0,1), left-tailed)",
        column_labels = ["Statistic", "Value", "P-value"],
        alignment = [:l, :r, :r])
    reject = r.pvalues[end] < 0.05    # ADF
    conclusion = reject ?
        "Reject H0 at 5% (ADF): evidence of cointegration" :
        "Fail to reject H0 (ADF): no evidence of cointegration"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::WesterlundResult{T}) where {T}
    has_boot = r.bootstrap > 0
    spec_data = Any[
        "H0"          "No error correction (no cointegration)";
        "H1"          "Error correction (cointegration)";
        "Deterministic" _regression_name(r.trend);
        "Regressors (k)" r.n_regressors;
        "Lags / Leads" string(r.lags, " / ", r.leads);
        "LR window"    r.lrwindow;
        "Bootstrap"    (has_boot ? "$(r.bootstrap) reps (seed $(r.seed))" : "none");
        "Units (N)"    r.n_units;
        "Time (T)"     r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Westerlund (2007) ECM Panel Cointegration Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    nstat = length(r.names)
    ncol = has_boot ? 4 : 3
    results_data = Matrix{Any}(undef, nstat, ncol)
    for s in 1:nstat
        results_data[s, 1] = r.names[s]
        results_data[s, 2] = string(round(r.statistics[s], digits=4), " ",
                                    _significance_stars(r.pvalues[s]))
        results_data[s, 3] = _format_pvalue(r.pvalues[s])
        if has_boot
            results_data[s, 4] = _format_pvalue(r.bootstrap_pvalues[s])
        end
    end
    labels = has_boot ? ["Statistic", "Z", "P-value", "Boot P"] :
                        ["Statistic", "Z", "P-value"]
    align = has_boot ? [:l, :r, :r, :r] : [:l, :r, :r]
    _pretty_table(io, results_data;
        title = "Statistics (Gt/Ga group-mean, Pt/Pa pooled; left-tailed)",
        column_labels = labels, alignment = align)
    reject = r.pvalues[1] < 0.05      # Gt
    conclusion = reject ?
        "Reject H0 at 5% (Gt): evidence of cointegration" :
        "Fail to reject H0 (Gt): no evidence of cointegration"
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::FisherJohansenResult{T}) where {T}
    combine_name = r.combine == :mw ? "Maddala-Wu (χ²(2N))" : "Choi (inv-normal Z)"
    spec_data = Any[
        "H0(r)"       "rank ≤ r (against rank > r)";
        "Combination" combine_name;
        "Deterministic" _regression_name(r.deterministic);
        "VAR lags (p)" r.lags;
        "Series (n)"   r.n_series;
        "Units (N)"    r.n_units;
        "Est. rank"    r.rank
    ]
    _pretty_table(io, spec_data;
        title = "Fisher-Type (Maddala-Wu / Choi) Panel Johansen Cointegration Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    nr = length(r.ranks)
    results_data = Matrix{Any}(undef, nr, 5)
    for j in 1:nr
        results_data[j, 1] = "r ≤ $(r.ranks[j])"
        results_data[j, 2] = round(r.trace_statistics[j], digits=4)
        results_data[j, 3] = string(_format_pvalue(r.trace_pvalues[j]), " ",
                                    _significance_stars(r.trace_pvalues[j]))
        results_data[j, 4] = round(r.max_statistics[j], digits=4)
        results_data[j, 5] = string(_format_pvalue(r.max_pvalues[j]), " ",
                                    _significance_stars(r.max_pvalues[j]))
    end
    _pretty_table(io, results_data;
        title = "Combined Statistics per Rank Hypothesis (primary: :$(r.combine))",
        column_labels = ["H0", "Trace stat", "Trace p", "Max stat", "Max p"],
        alignment = [:l, :r, :r, :r, :r])
    conc_data = Any[
        "Conclusion" "Estimated cointegration rank = $(r.rank) (first non-rejected trace test at 5%)";
        "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# Residual-based / parameter-stability cointegration tests (EV-11)
# =============================================================================

function Base.show(io::IO, r::EngleGrangerResult)
    spec_data = Any[
        "H₀" "No cointegration (unit root in residuals)";
        "H₁" "Cointegration";
        "Deterministic terms" _regression_name(r.regression);
        "I(1) series (N=k+1)" r.N;
        "Augmenting lags" r.lags;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Engle-Granger Two-Step Cointegration Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "ADF statistic (τ)" string(round(r.statistic, digits=4), " ", stars);
        "P-value (MacKinnon)" _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results", column_labels = ["", "Value"], alignment = [:l, :r])
    conc = r.pvalue < 0.01 ? "Reject H₀ at 1% (cointegration)" :
           r.pvalue < 0.05 ? "Reject H₀ at 5% (cointegration)" :
           r.pvalue < 0.10 ? "Reject H₀ at 10% (cointegration)" :
                             "Fail to reject H₀ (no cointegration)"
    _pretty_table(io, Any["Conclusion" conc; "Note" "*** p<0.01, ** p<0.05, * p<0.10"];
        column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::PhillipsOuliarisResult)
    spec_data = Any[
        "H₀" "No cointegration";
        "H₁" "Cointegration";
        "Deterministic terms" _regression_name(r.regression);
        "I(1) series (N=k+1)" r.N;
        "Kernel / bandwidth" string(r.kernel, " / ", round(r.bandwidth, digits=2));
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Phillips-Ouliaris Cointegration Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    results_data = Any[
        "Ẑ_t statistic" string(round(r.statistic, digits=4), " ", _significance_stars(r.pvalue));
        "Ẑ_t p-value (MacKinnon)" _format_pvalue(r.pvalue);
        "Ẑ_α statistic" string(round(r.z_alpha, digits=4), " ", _significance_stars(r.z_alpha_pvalue));
        "Ẑ_α p-value (MC table)" _format_pvalue(r.z_alpha_pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results", column_labels = ["", "Value"], alignment = [:l, :r])
    conc = r.pvalue < 0.05 ? "Reject H₀ at 5% (cointegration)" :
           r.pvalue < 0.10 ? "Reject H₀ at 10% (cointegration)" :
                             "Fail to reject H₀ (no cointegration)"
    _pretty_table(io, Any["Conclusion" conc; "Note" "*** p<0.01, ** p<0.05, * p<0.10"];
        column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::HansenInstabilityResult)
    spec_data = Any[
        "H₀" "Cointegration with stable coefficients";
        "H₁" "Parameter instability / no cointegration";
        "Deterministic terms" _regression_name(r.regression);
        "Parameters (p=d+k)" r.nparam;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Hansen (1992) Lc Parameter-Instability Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    results_data = Any[
        "Lc statistic" string(round(r.statistic, digits=4), " ", _significance_stars(r.pvalue));
        "P-value (MC table)" _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results", column_labels = ["", "Value"], alignment = [:l, :r])
    conc = r.pvalue < 0.01 ? "Reject H₀ at 1% (unstable / no cointegration)" :
           r.pvalue < 0.05 ? "Reject H₀ at 5% (unstable / no cointegration)" :
           r.pvalue < 0.10 ? "Reject H₀ at 10% (unstable / no cointegration)" :
                             "Fail to reject H₀ (stable cointegration)"
    _pretty_table(io, Any["Conclusion" conc; "Note" "*** p<0.01, ** p<0.05, * p<0.10"];
        column_labels=["",""], alignment=[:l,:l])
end

function Base.show(io::IO, r::ParkAddedResult)
    spec_data = Any[
        "H₀" "Genuine cointegration (I(0) errors)";
        "H₁" "Spurious regression (I(1) errors)";
        "Deterministic terms" _regression_name(r.regression);
        "Superfluous trends (q_add)" r.q_add;
        "Base trend order (p)" r.base_order;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "Park (1990) Added-Variables H(p,q) Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])
    results_data = Any[
        "H(p,q) statistic" string(round(r.statistic, digits=4), " ", _significance_stars(r.pvalue));
        "P-value (χ²($(r.q_add)))" _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results", column_labels = ["", "Value"], alignment = [:l, :r])
    conc = r.pvalue < 0.01 ? "Reject H₀ at 1% (spurious)" :
           r.pvalue < 0.05 ? "Reject H₀ at 5% (spurious)" :
           r.pvalue < 0.10 ? "Reject H₀ at 10% (spurious)" :
                             "Fail to reject H₀ (cointegration)"
    _pretty_table(io, Any["Conclusion" conc; "Note" "*** p<0.01, ** p<0.05, * p<0.10"];
        column_labels=["",""], alignment=[:l,:l])
end

# Dumitrescu-Hurlin panel Granger non-causality test (EV-24, #432)
# =============================================================================

function Base.show(io::IO, r::DumitrescuHurlinResult{T}) where {T}
    has_boot = r.bootstrap > 0 && isfinite(r.bootstrap_pvalue)
    spec_data = Any[
        "H₀"           "$(r.cause) does not Granger-cause $(r.effect) for any unit";
        "H₁"           "$(r.cause) Granger-causes $(r.effect) for some units";
        "Lag order (p)" r.p;
        "Units used (N)" r.N;
        "Eff. sample (T̄)" r.nobs;
        "Units skipped"  r.n_skipped;
        "Bootstrap"      (has_boot ? "$(r.bootstrap) reps (seed $(r.seed))" : "none")
    ]
    _pretty_table(io, spec_data;
        title = "Dumitrescu-Hurlin (2012) Panel Granger Non-Causality Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])

    results_data = Any[
        "W̄ (avg Wald, χ²(p))"  _fmt(r.Wbar);
        "Z̄ (asymptotic)"       string(_fmt(r.Zbar), " ", _significance_stars(r.Zbar_pvalue));
        "  P-value (Z̄)"        _format_pvalue(r.Zbar_pvalue);
        "Z̃ (small-T)"          string(_fmt(r.Ztilde), " ", _significance_stars(r.Ztilde_pvalue));
        "  P-value (Z̃)"        _format_pvalue(r.Ztilde_pvalue)
    ]
    if has_boot
        results_data = vcat(results_data,
            Any["Bootstrap P (Z̄)"  _format_pvalue(r.bootstrap_pvalue)])
    end
    _pretty_table(io, results_data;
        title = "Results (right-tailed)",
        column_labels = ["Statistic", "Value"], alignment = [:l, :r])

    pv = r.Ztilde_pvalue
    conclusion = if pv < 0.01
        "Reject H₀ at 1% — $(r.cause) Granger-causes $(r.effect) for some units"
    elseif pv < 0.05
        "Reject H₀ at 5% — $(r.cause) Granger-causes $(r.effect) for some units"
    elseif pv < 0.10
        "Reject H₀ at 10% — $(r.cause) Granger-causes $(r.effect) for some units"
    else
        "Fail to reject H₀ — no evidence that $(r.cause) Granger-causes $(r.effect)"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10 (on Z̃)"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# EDF goodness-of-fit battery (EV-26, #434)
# H₀: data follow the (specified/estimated) distribution; large statistic rejects.
# =============================================================================

function Base.show(io::IO, r::EDFTestResult{T}) where {T}
    param_mode = r.params == :estimate ? "estimated (ML)" : "specified"
    theta_str = isempty(r.theta) ? "—" :
        join([_fmt(t; digits=4) for t in r.theta], ", ")
    spec_data = Any[
        "H₀" "Data follow the $(_EDF_DIST_LABEL[r.dist]) distribution";
        "H₁" "Data do not follow the $(_EDF_DIST_LABEL[r.dist]) distribution";
        "Statistic" _EDF_TEST_LABEL[r.test];
        "Distribution" _EDF_DIST_LABEL[r.dist];
        "Parameters" "$param_mode: $theta_str";
        "Null case" r.case;
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = "EDF Goodness-of-Fit Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])

    has_p = isfinite(r.pvalue)
    stars = has_p ? _significance_stars(r.pvalue) : ""
    stat_label = r.statistic == r.raw_statistic ? "Statistic" : "Modified statistic"
    results_data = Any[
        stat_label            string(_fmt(r.statistic), " ", stars);
        "Raw EDF statistic"   _fmt(r.raw_statistic);
        "P-value"             (has_p ? _format_pvalue(r.pvalue) : "n/a (no null table)")
    ]
    _pretty_table(io, results_data;
        title = "Results (right-tailed)",
        column_labels = ["", "Value"], alignment = [:l, :r])

    if !isempty(r.critical_values)
        cv_data = Matrix{Any}(undef, 1, 3)
        cv_data[1, :] = [_fmt(r.critical_values[1]; digits=4),
                         _fmt(r.critical_values[5]; digits=4),
                         _fmt(r.critical_values[10]; digits=4)]
        _pretty_table(io, cv_data;
            title = "Critical Values",
            column_labels = ["1%", "5%", "10%"], alignment = :r)
    end

    conclusion = if !has_p
        "No published null table for this estimated family — inspect the statistic"
    elseif r.pvalue < 0.01
        "Reject H₀ at 1% — data do not follow the $(_EDF_DIST_LABEL[r.dist]) distribution"
    elseif r.pvalue < 0.05
        "Reject H₀ at 5% — data do not follow the $(_EDF_DIST_LABEL[r.dist]) distribution"
    elseif r.pvalue < 0.10
        "Reject H₀ at 10% — data do not follow the $(_EDF_DIST_LABEL[r.dist]) distribution"
    else
        "Fail to reject H₀ — no evidence against the $(_EDF_DIST_LABEL[r.dist]) fit"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# Equality-of-distribution + rank-correlation "Basic statistics" battery
# (EV-34, #442)
# =============================================================================

const _EQ_TEST_LABELS = Dict{Symbol,String}(
    :one_sample_t         => "One-Sample t-Test",
    :two_sample_t         => "Two-Sample t-Test (pooled)",
    :welch_t              => "Welch Two-Sample t-Test",
    :paired_t             => "Paired t-Test",
    :anova                => "One-Way ANOVA (F)",
    :welch_anova          => "Welch ANOVA (F)",
    :mann_whitney         => "Mann-Whitney U Test",
    :wilcoxon_signed_rank => "Wilcoxon Signed-Rank Test",
    :kruskal_wallis       => "Kruskal-Wallis H Test",
    :van_der_waerden      => "van der Waerden Normal-Scores Test",
    :median_chisq         => "Mood Median (χ²) Test",
    :variance_f           => "Two-Group Variance F-Test",
    :bartlett             => "Bartlett's Test",
    :levene               => "Levene's Test (center = mean)",
    :brown_forsythe       => "Brown-Forsythe Test (center = median)",
    :siegel_tukey         => "Siegel-Tukey Test",
)

const _EQ_STAT_LABELS = Dict{Symbol,String}(
    :one_sample_t         => "t statistic",
    :two_sample_t         => "t statistic",
    :welch_t              => "t statistic",
    :paired_t             => "t statistic",
    :anova                => "F statistic",
    :welch_anova          => "F statistic",
    :mann_whitney         => "U statistic",
    :wilcoxon_signed_rank => "V statistic",
    :kruskal_wallis       => "H statistic",
    :van_der_waerden      => "T statistic (χ²)",
    :median_chisq         => "χ² statistic",
    :variance_f           => "F statistic",
    :bartlett             => "χ² statistic",
    :levene               => "F statistic",
    :brown_forsythe       => "F statistic",
    :siegel_tukey         => "Rank-sum (group 1)",
)

# Tests whose primary statistic is on an F distribution (two df values).
_eq_is_ftest(name::Symbol) = name in (:anova, :welch_anova, :variance_f, :levene, :brown_forsythe)

function Base.show(io::IO, ::MIME"text/plain", r::EqualityTestResult{T}) where {T}
    stars = _significance_stars(r.pvalue)
    isf = _eq_is_ftest(r.test_name)
    df_str = isf ? string("(", _fmt(r.df1), ", ", _fmt(r.df2), ")") : _fmt(r.df1)
    spec_data = Any[
        "H₀"            "Groups share a common distribution / parameter";
        "H₁"            "Groups differ";
        "Method"        r.detail;
        "Groups"        r.n_groups;
        "Group sizes"   string(r.group_sizes);
        "Null"          (r.exact ? "exact" : "asymptotic approximation")
    ]
    _pretty_table(io, spec_data;
        title = "Equality Test: $(get(_EQ_TEST_LABELS, r.test_name, string(r.test_name)))",
        column_labels = ["Specification", ""], alignment = [:l, :r])

    results_data = Any[
        get(_EQ_STAT_LABELS, r.test_name, "Statistic")  string(_fmt(r.statistic), " ", stars);
        "Degrees of freedom"                             df_str;
        "P-value"                                        _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results", column_labels = ["", "Value"], alignment = [:l, :r])

    conclusion = if r.pvalue < 0.01
        "Reject H₀ at 1% significance level"
    elseif r.pvalue < 0.05
        "Reject H₀ at 5% significance level"
    elseif r.pvalue < 0.10
        "Reject H₀ at 10% significance level"
    else
        "Fail to reject H₀"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# Variance-ratio / random-walk tests (EV-27, #435)
# =============================================================================

function Base.show(io::IO, r::VarianceRatioResult{T}) where {T}
    has_boot = r.bootstrap > 0 && isfinite(r.cd_boot_pvalue)
    spec_data = Any[
        "H₀"              "Series is a random walk (VR(q)=1 ∀ q)";
        "H₁"              "Increments serially dependent (VR(q)≠1)";
        "Method"          (r.method === :wright ? "Lo-MacKinlay + Wright ranks/signs" : "Lo-MacKinlay");
        "Aggregations (q)" string(r.q);
        "Observations"    r.nobs;
        "Bootstrap"       (has_boot ? "$(r.bootstrap) reps, $(r.boot_weights) (seed $(r.seed))" : "none")
    ]
    _pretty_table(io, spec_data;
        title = "Variance-Ratio (Random-Walk) Test",
        column_labels = ["Specification", ""], alignment = [:l, :r])

    # Per-q Lo-MacKinlay table.
    nq = length(r.q)
    ncol = has_boot ? 6 : 5
    vr_data = Matrix{Any}(undef, nq, ncol)
    for i in 1:nq
        vr_data[i, 1] = r.q[i]
        vr_data[i, 2] = _fmt(r.vr[i])
        vr_data[i, 3] = string(_fmt(r.z[i]), " ", _significance_stars(r.z_pvalue[i]))
        vr_data[i, 4] = string(_fmt(r.z_star[i]), " ", _significance_stars(r.z_star_pvalue[i]))
        vr_data[i, 5] = _format_pvalue(r.z_star_pvalue[i])
        has_boot && (vr_data[i, 6] = _format_pvalue(r.z_star_boot_pvalue[i]))
    end
    vr_labels = has_boot ?
        ["q", "VR(q)", "Z(q)", "Z*(q)", "P(Z*)", "P(boot)"] :
        ["q", "VR(q)", "Z(q)", "Z*(q)", "P(Z*)"]
    _pretty_table(io, vr_data;
        title = "Lo-MacKinlay Variance Ratios",
        column_labels = vr_labels,
        alignment = [:r, :r, :r, :r, :r, :r][1:ncol])

    # Wright rank / sign statistics.
    if r.wright
        w_data = Matrix{Any}(undef, nq, 7)
        for i in 1:nq
            w_data[i, 1] = r.q[i]
            w_data[i, 2] = string(_fmt(r.R1[i]), " ", _significance_stars(r.R1_pvalue[i]))
            w_data[i, 3] = _format_pvalue(r.R1_pvalue[i])
            w_data[i, 4] = string(_fmt(r.R2[i]), " ", _significance_stars(r.R2_pvalue[i]))
            w_data[i, 5] = _format_pvalue(r.R2_pvalue[i])
            w_data[i, 6] = string(_fmt(r.S1[i]), " ", _significance_stars(r.S1_pvalue[i]))
            w_data[i, 7] = _format_pvalue(r.S1_pvalue[i])
        end
        _pretty_table(io, w_data;
            title = "Wright (2000) Rank / Sign Statistics (simulated null)",
            column_labels = ["q", "R1", "P(R1)", "R2", "P(R2)", "S1", "P(S1)"],
            alignment = [:r, :r, :r, :r, :r, :r, :r])
    end

    # Chow-Denning joint test.
    primary_asy = r.robust ? r.cd_star_pvalue : r.cd_pvalue
    joint_data = Any[
        "CD (homoskedastic)"  string(_fmt(r.cd_stat), " ", _significance_stars(r.cd_pvalue));
        "  P-value (SMM)"     _format_pvalue(r.cd_pvalue);
        "CD* (robust)"        string(_fmt(r.cd_star_stat), " ", _significance_stars(r.cd_star_pvalue));
        "  P-value (SMM)"     _format_pvalue(r.cd_star_pvalue)
    ]
    if has_boot
        joint_data = vcat(joint_data,
            Any["  P-value (wild boot)"  _format_pvalue(r.cd_boot_pvalue)])
    end
    _pretty_table(io, joint_data;
        title = "Chow-Denning Joint Test (max|Z|)",
        column_labels = ["Statistic", "Value"], alignment = [:l, :r])

    pv = has_boot ? r.cd_boot_pvalue : primary_asy
    branch = r.robust ? "robust" : "homoskedastic"
    conclusion = if pv < 0.01
        "Reject H₀ at 1% — series is not a random walk ($branch Chow-Denning)"
    elseif pv < 0.05
        "Reject H₀ at 5% — series is not a random walk ($branch Chow-Denning)"
    elseif pv < 0.10
        "Reject H₀ at 10% — series is not a random walk ($branch Chow-Denning)"
    else
        "Fail to reject H₀ — series consistent with a random walk"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10 (two-sided)"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# BDS independence test (Brock-Dechert-Scheinkman-LeBaron 1996) — EV-28 (#436)
# =============================================================================

function Base.show(io::IO, r::BDSResult{T}) where {T}
    has_boot = r.bootstrap > 0
    spec_data = Any[
        "H₀"                 "Observations are iid (independent)";
        "H₁"                 "Observations are not iid (nonlinear dependence / chaos)";
        "Embedding dims (m)" string(collect(r.m));
        "ε multipliers"      string("[", join(_fmt.(r.eps_frac), ", "), "] × sd");
        "Sample sd"          _fmt(r.sd);
        "Observations (T)"   r.nobs;
        "Bootstrap"          (has_boot ? "$(r.bootstrap) reps (seed $(r.seed))" : "none")
    ]
    _pretty_table(io, spec_data;
        title = "BDS Independence Test (Brock-Dechert-Scheinkman-LeBaron 1996)",
        column_labels = ["Specification", ""], alignment = [:l, :r])

    ncol = has_boot ? 6 : 5
    rows = length(r.m) * length(r.eps)
    data = Matrix{Any}(undef, rows, ncol)
    ri = 0
    for (im, m) in enumerate(r.m)
        for (je, eps) in enumerate(r.eps)
            ri += 1
            w = r.statistic[im, je]
            p = r.pvalue[im, je]
            data[ri, 1] = m
            data[ri, 2] = _fmt(eps)
            data[ri, 3] = _fmt(r.C[im, je])
            data[ri, 4] = isfinite(w) ? string(_fmt(w), " ", _significance_stars(p)) : "NaN"
            data[ri, 5] = isfinite(p) ? _format_pvalue(p) : "NaN"
            if has_boot
                pb = r.boot_pvalue[im, je]
                data[ri, 6] = isfinite(pb) ? _format_pvalue(pb) : "NaN"
            end
        end
    end
    labels = has_boot ? ["m", "ε", "C_m", "w (z)", "P-value", "Boot P"] :
                        ["m", "ε", "C_m", "w (z)", "P-value"]
    align = has_boot ? [:r, :r, :r, :r, :r, :r] : [:r, :r, :r, :r, :r]
    _pretty_table(io, data;
        title = "Results (two-sided N(0,1); one row per (m, ε))",
        column_labels = labels, alignment = align)

    pmin = minimum(r.pvalue)
    conclusion = pmin < 0.01 ? "Reject H₀ at 1% — series is not iid" :
                 pmin < 0.05 ? "Reject H₀ at 5% — series is not iid" :
                 pmin < 0.10 ? "Reject H₀ at 10% — series is not iid" :
                               "Fail to reject H₀ — no evidence against iid"
    note = r.small_sample ? "T<200: asymptotic p-values unreliable — prefer bootstrap" :
                            "*** p<0.01, ** p<0.05, * p<0.10"
    conc_data = Any["Conclusion" conclusion; "Note" note]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# Rank/association correlation tests (EV-34, #442)
# =============================================================================

function Base.show(io::IO, ::MIME"text/plain", r::CorTestResult{T}) where {T}
    stars = _significance_stars(r.pvalue)
    method_name = r.method === :pearson ? "Pearson product-moment" :
                  r.method === :spearman ? "Spearman rank" : "Kendall rank"
    coef_label = r.method === :pearson ? "Correlation r" :
                 r.method === :spearman ? "Correlation ρ" :
                 (r.exact ? "τ_a" : "τ_b")
    stat_label = r.method === :pearson ? "t statistic" :
                 r.method === :spearman ? "S statistic" :
                 (r.exact ? "T (concordant)" : "z statistic")
    spec_data = Any[
        "H₀"           "No association (coefficient = 0)";
        "H₁"           "Nonzero association";
        "Method"       r.detail;
        "Observations" r.n;
        "Null"         (r.exact ? "exact" : "asymptotic approximation")
    ]
    _pretty_table(io, spec_data;
        title = "Correlation Test: $(method_name)",
        column_labels = ["Specification", ""], alignment = [:l, :r])

    results_data = Any[
        coef_label   string(_fmt(r.estimate), " ", stars);
        stat_label   _fmt(r.statistic);
        "P-value"    _format_pvalue(r.pvalue)
    ]
    if r.method === :pearson && isfinite(r.ci_lower)
        results_data = vcat(results_data,
            Any["95% CI" string("[", _fmt(r.ci_lower), ", ", _fmt(r.ci_upper), "]")])
    end
    _pretty_table(io, results_data;
        title = "Results", column_labels = ["", "Value"], alignment = [:l, :r])

    conclusion = if r.pvalue < 0.01
        "Reject H₀ at 1% — significant association"
    elseif r.pvalue < 0.05
        "Reject H₀ at 5% — significant association"
    elseif r.pvalue < 0.10
        "Reject H₀ at 10% — marginal association"
    else
        "Fail to reject H₀ — no significant association"
    end
    conc_data = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc_data; column_labels=["",""], alignment=[:l,:l])
end

# Plain 2-arg forwarders so report()/print render the same table (EV-34).
Base.show(io::IO, r::EqualityTestResult) = show(io, MIME"text/plain"(), r)
Base.show(io::IO, r::CorTestResult) = show(io, MIME"text/plain"(), r)

# =============================================================================
# Explosive / rational-bubble detection (EV-30, #438). RIGHT-TAILED sup-ADF:
# reject the unit-root null for a LARGE statistic (stat > CV), upper quantiles.
# Unique trailing section (episode roster) — do not rely on shared trailing
# lines when cherry-picking this branch.
# =============================================================================
function Base.show(io::IO, r::BubbleResult)
    kind_name = r.kind == :sadf ? "Supremum ADF (SADF)" : "Generalized Supremum ADF (GSADF)"
    provenance = r.kind == :sadf ? "Phillips-Wu-Yu (2011)" : "Phillips-Shi-Yu (2015)"
    cv_label = r.cv_method == :wildboot ? "wild bootstrap (Phillips-Shi 2020)" : "asymptotic MC"
    spec_data = Any[
        "H₀" "Unit root (no explosive behaviour)";
        "H₁" "Mildly explosive root (bubble) — right-tailed";
        "Statistic" kind_name;
        "Min. window r₀" string(round(r.r0, digits=4));
        "Augmenting lags" r.adflag;
        "Critical values" string(cv_label, ", ", r.mc_reps, " reps");
        "Observations" r.nobs
    ]
    _pretty_table(io, spec_data;
        title = string(kind_name, " Bubble Test — ", provenance),
        column_labels = ["Specification", ""],
        alignment = [:l, :r],
    )
    stars = _significance_stars(r.pvalue)
    results_data = Any[
        "Sup statistic" string(round(r.statistic, digits=4), " ", stars);
        "P-value" _format_pvalue(r.pvalue)
    ]
    _pretty_table(io, results_data;
        title = "Results", column_labels = ["", "Value"], alignment = [:l, :r])
    cv_data = Matrix{Any}(undef, 1, 3)
    cv_data[1, :] = [round(r.critical_values[10], digits=3),
                     round(r.critical_values[5], digits=3),
                     round(r.critical_values[1], digits=3)]
    _pretty_table(io, cv_data;
        title = "Right-Tailed Critical Values (upper quantiles)",
        column_labels = ["10%", "5%", "1%"], alignment = :r)
    decision = if r.statistic > r.critical_values[1]
        "Reject H₀ at 1% — explosive behaviour detected"
    elseif r.statistic > r.critical_values[5]
        "Reject H₀ at 5% — explosive behaviour detected"
    elseif r.statistic > r.critical_values[10]
        "Reject H₀ at 10% — explosive behaviour detected"
    else
        "Fail to reject H₀ — no evidence of a bubble"
    end
    dec_data = Any["Decision" decision; "Note" "reject when statistic > CV (right-tailed)"]
    _pretty_table(io, dec_data; column_labels=["",""], alignment=[:l,:l])

    # Unique trailing section: stamped bubble episode roster.
    if isempty(r.episodes)
        ep_data = Any["No bubble episodes stamped" string("(min duration ", max(1, ceil(Int, log(r.nobs))), " obs)")]
        _pretty_table(io, ep_data; title="Date-Stamped Bubble Episodes",
            column_labels=["",""], alignment=[:l,:r])
    else
        ep_rows = Matrix{Any}(undef, length(r.episodes), 3)
        for (i, (s, e)) in enumerate(r.episodes)
            ep_rows[i, 1] = i
            ep_rows[i, 2] = string(s, " – ", e)
            ep_rows[i, 3] = e - s + 1
        end
        _pretty_table(io, ep_rows;
            title = string("Date-Stamped Bubble Episodes (", length(r.episodes), ")"),
            column_labels = ["#", "Index range", "Duration"],
            alignment = [:r, :c, :r])
    end
    return nothing
end
