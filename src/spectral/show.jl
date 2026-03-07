# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
Display methods for spectral analysis and diagnostic test results.
"""

# =============================================================================
# ACFResult
# =============================================================================

function Base.show(io::IO, r::ACFResult{T}) where {T}
    # Determine what we have
    has_acf = any(!iszero, r.acf)
    has_pacf = any(!iszero, r.pacf)
    has_ccf = r.ccf !== nothing
    has_q = any(!iszero, r.q_stats)

    if has_ccf
        # CCF display
        title = "Cross-Correlation Function (n=$(r.nobs))"
        n_lags = length(r.lags)
        n_show = min(n_lags, 30)
        # Show representative lags
        if n_show < n_lags
            indices = unique(vcat(1:min(5, n_lags),
                                  div(n_lags, 2)-2:div(n_lags, 2)+2,
                                  max(1, n_lags-4):n_lags))
            indices = sort(filter(i -> 1 <= i <= n_lags, indices))
        else
            indices = 1:n_lags
        end
        data = Matrix{Any}(undef, length(indices), 2)
        for (row, i) in enumerate(indices)
            data[row, 1] = r.lags[i]
            data[row, 2] = _fmt(r.ccf[i])
        end
        _pretty_table(io, data;
            title = title,
            column_labels = ["Lag", "CCF"],
            alignment = [:r, :r],
        )
        note = Any["CI ($(round(Int, 100*(1-2*ccdf(Normal(), r.ci * sqrt(T(r.nobs))))))%)" "$(string(round(r.ci, digits=4)))"]
        _pretty_table(io, note; column_labels=["",""], alignment=[:l,:r])
        return
    end

    # ACF/PACF display (Stata-style correlogram)
    if has_acf && has_pacf
        title = "Correlogram (n=$(r.nobs))"
    elseif has_acf
        title = "Autocorrelation Function (n=$(r.nobs))"
    else
        title = "Partial Autocorrelation Function (n=$(r.nobs))"
    end

    n_lags = length(r.lags)
    n_show = min(n_lags, 40)

    if has_acf && has_pacf && has_q
        data = Matrix{Any}(undef, n_show, 5)
        for i in 1:n_show
            data[i, 1] = r.lags[i]
            data[i, 2] = _fmt(r.acf[i])
            data[i, 3] = _fmt(r.pacf[i])
            data[i, 4] = _fmt(r.q_stats[i]; digits=2)
            data[i, 5] = _format_pvalue(r.q_pvalues[i])
        end
        _pretty_table(io, data;
            title = title,
            column_labels = ["Lag", "AC", "PAC", "Q-Stat", "Prob"],
            alignment = [:r, :r, :r, :r, :r],
        )
    elseif has_acf && has_q
        data = Matrix{Any}(undef, n_show, 4)
        for i in 1:n_show
            data[i, 1] = r.lags[i]
            data[i, 2] = _fmt(r.acf[i])
            data[i, 3] = _fmt(r.q_stats[i]; digits=2)
            data[i, 4] = _format_pvalue(r.q_pvalues[i])
        end
        _pretty_table(io, data;
            title = title,
            column_labels = ["Lag", "AC", "Q-Stat", "Prob"],
            alignment = [:r, :r, :r, :r],
        )
    elseif has_pacf
        data = Matrix{Any}(undef, n_show, 2)
        for i in 1:n_show
            data[i, 1] = r.lags[i]
            data[i, 2] = _fmt(r.pacf[i])
        end
        _pretty_table(io, data;
            title = title,
            column_labels = ["Lag", "PAC"],
            alignment = [:r, :r],
        )
    else
        data = Matrix{Any}(undef, n_show, 2)
        for i in 1:n_show
            data[i, 1] = r.lags[i]
            data[i, 2] = _fmt(r.acf[i])
        end
        _pretty_table(io, data;
            title = title,
            column_labels = ["Lag", "AC"],
            alignment = [:r, :r],
        )
    end

    note_data = Any["CI (95%)" "$(string(round(r.ci, digits=4)))"]
    _pretty_table(io, note_data; column_labels=["",""], alignment=[:l,:r])
end

# =============================================================================
# SpectralDensityResult
# =============================================================================

function Base.show(io::IO, r::SpectralDensityResult{T}) where {T}
    n_freq = length(r.freq)
    # Show summary stats
    spec = Any[
        "Method"       string(r.method);
        "Observations" r.nobs;
        "Frequencies"  n_freq;
        "Bandwidth"    _fmt(r.bandwidth);
        "Peak density" _fmt(maximum(r.density));
        "Peak freq"    _fmt(r.freq[argmax(r.density)])
    ]
    _pretty_table(io, spec;
        title = "Spectral Density Estimate",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# CrossSpectrumResult
# =============================================================================

function Base.show(io::IO, r::CrossSpectrumResult{T}) where {T}
    n_freq = length(r.freq)
    peak_coh_idx = argmax(r.coherence)
    spec = Any[
        "Observations"   r.nobs;
        "Frequencies"    n_freq;
        "Max coherence"  _fmt(r.coherence[peak_coh_idx]);
        "at frequency"   _fmt(r.freq[peak_coh_idx])
    ]
    _pretty_table(io, spec;
        title = "Cross-Spectral Analysis",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# TransferFunctionResult
# =============================================================================

function Base.show(io::IO, r::TransferFunctionResult{T}) where {T}
    n_freq = length(r.freq)
    spec = Any[
        "Filter"         string(r.filter);
        "Frequencies"    n_freq;
        "Max gain"       _fmt(maximum(r.gain));
        "Min gain"       _fmt(minimum(r.gain))
    ]
    _pretty_table(io, spec;
        title = "Transfer Function: $(r.filter)",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

# =============================================================================
# LjungBoxResult
# =============================================================================

function Base.show(io::IO, r::LjungBoxResult{T}) where {T}
    stars = _significance_stars(r.pvalue)
    reject_5 = r.pvalue < 0.05
    conclusion = reject_5 ? "Reject H₀ (serial correlation detected)" :
                            "Fail to reject H₀ (no evidence of serial correlation)"
    data = Any[
        "H₀"            "No serial correlation up to lag $(r.lags)";
        "H₁"            "Serial correlation present";
        "Q-Statistic"   string(_fmt(r.statistic), " ", stars);
        "P-value"        _format_pvalue(r.pvalue);
        "Lags"           r.lags;
        "DOF"            r.df;
        "Observations"   r.nobs
    ]
    _pretty_table(io, data;
        title = "Ljung-Box Q Test",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    conc = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# BoxPierceResult
# =============================================================================

function Base.show(io::IO, r::BoxPierceResult{T}) where {T}
    stars = _significance_stars(r.pvalue)
    reject_5 = r.pvalue < 0.05
    conclusion = reject_5 ? "Reject H₀ (serial correlation detected)" :
                            "Fail to reject H₀ (no evidence of serial correlation)"
    data = Any[
        "H₀"            "No serial correlation up to lag $(r.lags)";
        "H₁"            "Serial correlation present";
        "Q₀-Statistic"  string(_fmt(r.statistic), " ", stars);
        "P-value"        _format_pvalue(r.pvalue);
        "Lags"           r.lags;
        "DOF"            r.df;
        "Observations"   r.nobs
    ]
    _pretty_table(io, data;
        title = "Box-Pierce Q₀ Test",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    conc = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# DurbinWatsonResult
# =============================================================================

function Base.show(io::IO, r::DurbinWatsonResult{T}) where {T}
    stars = _significance_stars(r.pvalue)
    if r.statistic < 1.5
        interp = "Positive autocorrelation (DW < 2)"
    elseif r.statistic > 2.5
        interp = "Negative autocorrelation (DW > 2)"
    else
        interp = "No strong evidence of autocorrelation (DW ≈ 2)"
    end
    data = Any[
        "H₀"            "No first-order autocorrelation";
        "DW Statistic"   string(_fmt(r.statistic), " ", stars);
        "P-value"        _format_pvalue(r.pvalue);
        "Observations"   r.nobs
    ]
    _pretty_table(io, data;
        title = "Durbin-Watson Test",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    conc = Any["Interpretation" interp; "Note" "DW ∈ [0,4]; 2 = no autocorrelation"]
    _pretty_table(io, conc; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# FisherTestResult
# =============================================================================

function Base.show(io::IO, r::FisherTestResult{T}) where {T}
    stars = _significance_stars(r.pvalue)
    reject_5 = r.pvalue < 0.05
    conclusion = reject_5 ? "Reject H₀ (periodic component detected)" :
                            "Fail to reject H₀ (no evidence of hidden periodicity)"
    data = Any[
        "H₀"            "No hidden periodicity (white noise)";
        "H₁"            "Dominant periodic component present";
        "Fisher's g"     string(_fmt(r.statistic), " ", stars);
        "P-value"        _format_pvalue(r.pvalue);
        "Peak frequency" _fmt(r.peak_freq);
        "Observations"   r.nobs
    ]
    _pretty_table(io, data;
        title = "Fisher's Test for Periodicity",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    conc = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc; column_labels=["",""], alignment=[:l,:l])
end

# =============================================================================
# BartlettWhiteNoiseResult
# =============================================================================

function Base.show(io::IO, r::BartlettWhiteNoiseResult{T}) where {T}
    stars = _significance_stars(r.pvalue)
    reject_5 = r.pvalue < 0.05
    conclusion = reject_5 ? "Reject H₀ (series is NOT white noise)" :
                            "Fail to reject H₀ (consistent with white noise)"
    data = Any[
        "H₀"            "Series is white noise";
        "H₁"            "Series is not white noise";
        "KS Statistic"  string(_fmt(r.statistic), " ", stars);
        "P-value"        _format_pvalue(r.pvalue);
        "Observations"   r.nobs
    ]
    _pretty_table(io, data;
        title = "Bartlett's White Noise Test",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    conc = Any["Conclusion" conclusion; "Note" "*** p<0.01, ** p<0.05, * p<0.10"]
    _pretty_table(io, conc; column_labels=["",""], alignment=[:l,:l])
end
