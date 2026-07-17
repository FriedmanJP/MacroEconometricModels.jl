# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# =============================================================================
# Nonparametric module — compact displays (EV-33, #441)
# =============================================================================

const _NP_KERNEL_LABEL = Dict(
    :gaussian => "Gaussian", :epanechnikov => "Epanechnikov",
    :triangular => "Triangular", :uniform => "Uniform (rectangular)",
)
const _NP_BW_LABEL = Dict(
    :silverman => "Silverman (bw.nrd0)", :sj => "Sheather–Jones plug-in",
    :cv => "Leave-one-out CV", :rot => "Rule of thumb", :user => "User-specified",
)
const _NP_METHOD_LABEL = Dict(
    :nw => "Nadaraya–Watson (local constant)",
    :ll => "Local linear", :lp => "Local polynomial",
)

function Base.show(io::IO, r::KernelDensity{T}) where {T}
    spec = Any[
        "Kernel"        get(_NP_KERNEL_LABEL, r.kernel, string(r.kernel));
        "Bandwidth"     _fmt(r.bandwidth);
        "Bandwidth rule" get(_NP_BW_LABEL, r.bw_method, string(r.bw_method));
        "Observations"  r.nobs;
        "Grid points"   length(r.x);
        "Peak density"  _fmt(maximum(r.density));
        "Peak at x"     _fmt(r.x[argmax(r.density)])
    ]
    _pretty_table(io, spec;
        title = "Kernel Density Estimate",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::KernelRegression{T}) where {T}
    deglabel = r.method === :lp ? " (degree $(r.degree))" : ""
    spec = Any[
        "Method"        (get(_NP_METHOD_LABEL, r.method, string(r.method)) * deglabel);
        "Kernel"        get(_NP_KERNEL_LABEL, r.kernel, string(r.kernel));
        "Bandwidth"     _fmt(r.bandwidth);
        "Bandwidth rule" get(_NP_BW_LABEL, r.bw_method, string(r.bw_method));
        "Observations"  r.nobs;
        "Residual σ̂²"   _fmt(r.sigma2);
        "Mean |SE|"     _fmt(sum(r.se) / length(r.se))
    ]
    _pretty_table(io, spec;
        title = "Nonparametric Regression",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::LowessFit{T}) where {T}
    resid = r.ydata .- r.fitted
    rss = sum(abs2, resid)
    spec = Any[
        "Smoother"      "LOWESS (Cleveland 1979)";
        "Span (f)"      _fmt(r.span);
        "Iterations"    r.iter;
        "Window size"   max(min(Int(floor(r.span * r.nobs + 1e-7)), r.nobs), 2);
        "Observations"  r.nobs;
        "Residual SS"   _fmt(rss)
    ]
    _pretty_table(io, spec;
        title = "LOWESS Smoother",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end
