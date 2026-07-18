# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

"""
plot_result method for BayesianDSGE (relocated out of models.jl in the PLT
plotting overhaul so the Wave-2 MCMC lane owns this file). PLT-05: the posterior
mean is now drawn as a real vertical reference line (axis:"x"), matching the
docstring. PLT-19: the posterior KDE is computed by the shared `_kde_line` helper
(helpers.jl), no longer inline.
"""

# =============================================================================
# BayesianDSGE — Prior vs Posterior Density Plots
# =============================================================================

"""
    plot_result(result::BayesianDSGE; title="", save_path=nothing, ncols=0)

Plot prior vs posterior density for each estimated parameter. Each panel shows
the prior density curve (dashed) and a kernel density estimate of the posterior
draws (solid), with a vertical line at the posterior mean.
"""
function plot_result(result::BayesianDSGE{T};
                     title::String="", ncols::Int=0,
                     save_path::Union{String,Nothing}=nothing) where {T}
    n_params = length(result.param_names)
    panels = _PanelSpec[]

    for i in 1:n_params
        pn = string(result.param_names[i])
        draws = result.theta_draws[:, i]
        d = result.priors.distributions[i]
        post_mean = mean(draws)

        # KDE of the posterior draws via the shared Silverman-bandwidth helper
        # (PLT-19 — A5, no inline statistics). Prior is evaluated on the SAME grid.
        xs, kde_vals = _kde_line(draws)
        n_grid = length(xs)

        # Prior density at same grid points
        prior_vals = zeros(Float64, n_grid)
        for (gi, xg) in enumerate(xs)
            try
                pv = pdf(d, xg)
                prior_vals[gi] = isfinite(pv) ? pv : zero(T)
            catch
                prior_vals[gi] = zero(T)
            end
        end

        # Build data JSON
        rows = Vector{Pair{String,String}}[]
        for gi in 1:n_grid
            push!(rows, [
                "x" => _json(xs[gi]),
                "post" => _json(kde_vals[gi]),
                "prior" => _json(prior_vals[gi])
            ])
        end
        data_json = _json_array_of_objects(rows)

        id = _next_plot_id("bayes")

        # Use line chart with prior (dashed) and posterior (solid)
        s_json = "[{\"name\":\"Posterior\",\"color\":\"$(_PLOT_COLORS[1])\",\"key\":\"post\",\"dash\":\"\"}," *
                 "{\"name\":\"Prior\",\"color\":\"$(_PLOT_COLORS[2])\",\"key\":\"prior\",\"dash\":\"6,3\"}]"

        # PLT-05: vertical reference line at the posterior mean (axis:"x"), matching
        # the docstring. `post_mean` lies inside the KDE x-grid, so x(post_mean) is
        # in range.
        refs_json = "[{\"value\":$(_json(post_mean)),\"axis\":\"x\",\"color\":\"#d62728\",\"dash\":\"4,3\"}]"

        js = _render_line_js(id, data_json, s_json;
                             ref_lines_json=refs_json, xlabel=pn, ylabel="Density")

        push!(panels, _PanelSpec(id, pn, js))
    end

    if isempty(title)
        title = "Bayesian DSGE — Prior vs Posterior"
    end

    p = _make_plot(panels; title=title, ncols=ncols)
    save_path !== nothing && save_plot(p, save_path)
    p
end
