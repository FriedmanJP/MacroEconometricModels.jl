# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
#
# MacroEconometricModels.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MacroEconometricModels.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MacroEconometricModels.jl. If not, see <https://www.gnu.org/licenses/>.

# =============================================================================
# Nowcasting Display Methods
# =============================================================================

function Base.show(io::IO, m::NowcastDFM{T}) where {T}
    T_obs, N = size(m.data)
    n_nan = count(isnan, m.data)
    n_filled = count(isnan, m.data) - count(isnan, m.X_sm)

    spec_data = Any[
        "Method"        "Dynamic Factor Model (EM)";
        "Variables"     "$N ($(m.nM) monthly, $(m.nQ) quarterly)";
        "Observations"  T_obs;
        "Factors"       m.r;
        "Factor lags"   m.p;
        "Idiosyncratic" string(m.idio);
        "Blocks"        size(m.blocks, 2);
        "EM iterations" m.n_iter;
        "Log-likelihood" _fmt(m.loglik);
        "Missing values" n_nan;
    ]
    _pretty_table(io, spec_data;
        title = "DFM Nowcasting",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, m::NowcastBVAR{T}) where {T}
    T_obs, N = size(m.data)
    n_nan = count(isnan, m.data)

    spec_data = Any[
        "Method"           "Large BVAR (GLP prior)";
        "Variables"        "$N ($(m.nM) monthly, $(m.nQ) quarterly)";
        "Observations"     T_obs;
        "Lags"             m.lags;
        "Log-likelihood"   _fmt(m.loglik);
        "Lambda (shrinkage)" _fmt(m.lambda);
        "Theta (cross-var)"  _fmt(m.theta);
        "Miu (unit root)"    _fmt(m.miu);
        "Alpha (co-persist)" _fmt(m.alpha);
        "Missing values"   n_nan;
    ]
    _pretty_table(io, spec_data;
        title = "BVAR Nowcasting",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, m::NowcastBridge{T}) where {T}
    T_obs, N = size(m.data)
    n_quarters = length(m.Y_nowcast)
    last_nowcast = m.Y_nowcast[n_quarters]

    spec_data = Any[
        "Method"            "Bridge Equation Combination";
        "Variables"         "$N ($(m.nM) monthly, $(m.nQ) quarterly)";
        "Observations"      T_obs;
        "Bridge equations"  m.n_equations;
        "Monthly lags"      m.lagM;
        "Quarterly lags"    m.lagQ;
        "AR lags"           m.lagY;
        "Current nowcast"   isnan(last_nowcast) ? "N/A" : string(_fmt(last_nowcast));
    ]
    _pretty_table(io, spec_data;
        title = "Bridge Equation Nowcasting",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, r::NowcastResult{T}) where {T}
    method_str = r.method == :dfm ? "DFM" : r.method == :bvar ? "BVAR" : "Bridge"

    spec_data = Any[
        "Method"           method_str;
        "Target variable"  r.target_index;
        "Current nowcast"  _fmt(r.nowcast);
        "Next forecast"    _fmt(r.forecast);
    ]
    _pretty_table(io, spec_data;
        title = "Nowcast Result",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
end

function Base.show(io::IO, n::NowcastNews{T}) where {T}
    total = n.new_nowcast - n.old_nowcast
    n_releases = length(n.impact_news)

    spec_data = Any[
        "Old nowcast"       _fmt(n.old_nowcast);
        "New nowcast"       _fmt(n.new_nowcast);
        "Total revision"    _fmt(total);
        "News impact"       _fmt(sum(n.impact_news));
        "Revision impact"   _fmt(n.impact_revision);
        "Reestimation"      _fmt(n.impact_reestimation);
        "New releases"      n_releases;
    ]
    _pretty_table(io, spec_data;
        title = "Nowcast News Decomposition",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )

    # Show top contributors if there are releases
    if n_releases > 0
        n_show = min(n_releases, 10)
        sorted_idx = sortperm(abs.(n.impact_news), rev=true)[1:n_show]
        contrib_data = Matrix{Any}(undef, n_show, 2)
        for (i, idx) in enumerate(sorted_idx)
            contrib_data[i, 1] = idx <= length(n.variable_names) ? n.variable_names[idx] : "Release $idx"
            contrib_data[i, 2] = _fmt(n.impact_news[idx])
        end
        _pretty_table(io, contrib_data;
            title = "Top Contributors",
            column_labels = ["Release", "Impact"],
            alignment = [:l, :r],
        )
    end
end

"""
    report(m::AbstractNowcastModel)

Print comprehensive nowcasting model summary.
"""
report(m::AbstractNowcastModel) = show(stdout, m)
report(r::NowcastResult) = show(stdout, r)
report(n::NowcastNews) = show(stdout, n)
