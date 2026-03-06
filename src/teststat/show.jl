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
    cv_data[1, :] = [round(r.critical_values[10], digits=3),
                     round(r.critical_values[5], digits=3),
                     round(r.critical_values[1], digits=3)]
    _pretty_table(io, cv_data;
        title = "Critical Values",
        column_labels = ["10%", "5%", "1%"],
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
        pval_str = pval < 0.001 ? "<0.001" : string(round(pval, digits=4))
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
        pval_str = pval < 0.001 ? "<0.001" : string(round(pval, digits=4))
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
