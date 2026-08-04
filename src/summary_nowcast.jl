# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

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
        "Idiosyncratic" _label(m.idio);
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
    # The headline the model exists to produce (default target = last variable). (S4/T168)
    _pretty_table(io, Any["Current nowcast" _fmt(m.X_sm[end, end])];
        title = "Nowcast", column_labels = ["", ""], alignment = [:l, :r])
end

function Base.show(io::IO, m::NowcastBVAR{T}) where {T}
    T_obs, N = size(m.data)
    n_nan = count(isnan, m.data)

    lit = m.prior == :litterman
    spec_data = Any[
        "Method"           (lit ? "Large BVAR (Litterman prior)" : "Large BVAR (GLP prior)");
        "Variables"        "$N ($(m.nM) monthly, $(m.nQ) quarterly)";
        "Observations"     T_obs;
        "Lags"             m.lags;
        "Log marg. lik."   _fmt(m.loglik);
        "Lambda (shrinkage)" _fmt(m.lambda);
        "Theta (lag decay)"  _fmt(m.theta);
        "Miu (unit root)"    _fmt(m.miu);
        "Alpha (co-persist)" _fmt(m.alpha);
        # Only the non-conjugate prior has a cross-variable knob; under the conjugate NIW
        # prior the own/cross ratio is fixed at √(Σ_mm/Σ_jj), so there is no value to show.
        "Theta_cross (cross-var)" (lit ? _fmt(m.theta_cross) : "— (fixed by Σ⊗V)");
        "Missing values"   n_nan;
    ]
    _pretty_table(io, spec_data;
        title = "BVAR Nowcasting",
        column_labels = ["", ""],
        alignment = [:l, :r],
    )
    # The headline the model exists to produce (default target = last variable). (S4/T168)
    _pretty_table(io, Any["Current nowcast" _fmt(m.X_sm[end, end])];
        title = "Nowcast", column_labels = ["", ""], alignment = [:l, :r])
    # Sanity flag: a boundary-hit optimizer parks a hyperparameter at exp(±5) — never
    # present it bare. Name which ones and which edge: the ceiling (exp(5) ≈ 148.4) and the
    # floor (exp(-5) ≈ 0.0067) mean opposite things, and the old text said "λ = exp(5)"
    # even when the hit was a floor on a different hyperparameter. (B4/T173, #602)
    if !m.converged
        hi, lo = exp(one(T) * 5), exp(-one(T) * 5)
        atedge(v, e) = isfinite(v) && abs(log(v) - log(e)) < T(1e-3)
        pairs = ("lambda" => m.lambda, "theta" => m.theta, "miu" => m.miu,
                 "alpha" => m.alpha, "theta_cross" => m.theta_cross)
        ceil_hits = [n for (n, v) in pairs if atedge(v, hi)]
        floor_hits = [n for (n, v) in pairs if atedge(v, lo)]
        parts = String[]
        isempty(ceil_hits) || push!(parts, "$(join(ceil_hits, ", ")) at the ceiling exp(5) ≈ 148.4")
        isempty(floor_hits) || push!(parts, "$(join(floor_hits, ", ")) at the floor exp(-5) ≈ 0.0067")
        detail = isempty(parts) ? "a hyperparameter reached the edge" : join(parts, "; ")
        println(io, "WARNING: hyperparameters hit the |log-param| ≤ 5 box — $detail. The " *
                    "marginal likelihood was still improving, so these are truncation " *
                    "points, not optima; treat them as bounds, not estimates.")
        if "theta_cross" in floor_hits
            println(io, "NOTE: theta_cross at the floor means the criterion wants " *
                        "cross-variable lags shrunk essentially to zero — the panel " *
                        "supports little more than N separate autoregressions.")
        end
    end
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
