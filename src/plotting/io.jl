# io.jl — plotting recipes for Input-Output results (inline D3.js)

"""
    plot_result(r::ExtractionResult; title="", save_path=nothing) -> PlotOutput

Bar chart of per-sector gross-output losses from hypothetical extraction.
"""
function plot_result(r::ExtractionResult; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_extract")
    labs = ["sector $(i)" for i in 1:length(r.sector_loss)]
    rows = [Pair{String,String}["x" => _json(labs[i]), "Loss" => _json(r.sector_loss[i])]
            for i in 1:length(r.sector_loss)]
    data_json = _json_array_of_objects(rows)
    series_json = "[" * _json_obj(Pair{String,String}[
        "key" => _json("Loss"), "name" => _json("Output loss"),
        "color" => _json(_PLOT_COLORS[1])]) * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Sector", ylabel="Output loss")
    ttl = isempty(title) ?
        "Extraction loss (mode=$(r.mode), total=$(_fmt(r.total_loss)))" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::PriceModelResult; title="", save_path=nothing) -> PlotOutput

Bar chart of sectoral price changes ``Δp`` from the cost-push price model.
"""
function plot_result(r::PriceModelResult; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_price")
    rows = [Pair{String,String}["x" => _json(r.sectors[i]), "Δp" => _json(r.dp[i])]
            for i in 1:length(r.dp)]
    data_json = _json_array_of_objects(rows)
    series_json = "[" * _json_obj(Pair{String,String}[
        "key" => _json("Δp"), "name" => _json("Δp"),
        "color" => _json(_PLOT_COLORS[1])]) * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Sector", ylabel="Δp")
    ttl = isempty(title) ? "Price model ($(r.mode))" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::ImpactResult; title="", save_path=nothing) -> PlotOutput

Bar chart of per-sector impacts from a final-demand scenario.
"""
function plot_result(r::ImpactResult; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_impact")
    rows = [Pair{String,String}["x" => _json(r.sectors[i]),
                                "Impact" => _json(r.by_sector[i])]
            for i in 1:length(r.by_sector)]
    data_json = _json_array_of_objects(rows)
    series_json = "[" * _json_obj(Pair{String,String}[
        "key" => _json("Impact"), "name" => _json("Impact"),
        "color" => _json(_PLOT_COLORS[1])]) * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Sector", ylabel="Impact")
    ttl = isempty(title) ?
        "Impact ($(r.type) $(r.kind); total=$(_fmt(r.total)))" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::NetworkStatsResult; title="", save_path=nothing) -> PlotOutput

Bar chart of Domar weights from network statistics.
"""
function plot_result(r::NetworkStatsResult; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_network")
    rows = [Pair{String,String}["x" => _json(r.sectors[i]),
                                "Domar" => _json(r.domar[i])]
            for i in 1:length(r.domar)]
    data_json = _json_array_of_objects(rows)
    series_json = "[" * _json_obj(Pair{String,String}[
        "key" => _json("Domar"), "name" => _json("Domar λ"),
        "color" => _json(_PLOT_COLORS[1])]) * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Sector", ylabel="Domar weight")
    ttl = isempty(title) ?
        "NetworkStats Domar (HHI=$(_fmt(r.herfindahl)))" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(m::IOMultipliers; title="", save_path=nothing) -> PlotOutput

Bar chart of sectoral multipliers.
"""
function plot_result(m::IOMultipliers; title::String="", save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_mult")
    rows = [Pair{String,String}["x" => _json(m.sectors[i]), "Multiplier" => _json(m.values[i])]
            for i in 1:length(m.values)]
    data_json = _json_array_of_objects(rows)
    series_json = "[" * _json_obj(Pair{String,String}[
        "key" => _json("Multiplier"), "name" => _json("Multiplier"),
        "color" => _json(_PLOT_COLORS[1])]) * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Sector", ylabel="Multiplier")
    ttl = isempty(title) ? "$(m.type) $(m.kind) multipliers" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(lk::LinkageResult; title="Rasmussen linkages", save_path=nothing) -> PlotOutput

Scatter of Rasmussen power-of-dispersion (`U_i`) vs sensitivity-of-dispersion
(`U_j`) with reference lines at 1, dividing sectors into the key-sector quadrants.
"""
function plot_result(lk::LinkageResult; title::String="Rasmussen linkages",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_rasmussen")
    rows = [Pair{String,String}["x" => _json(lk.Ui[i]), "y" => _json(lk.Uj[i]),
                                "group" => _json(lk.sectors[i])]
            for i in 1:length(lk.sectors)]
    data_json = _json_array_of_objects(rows)
    grows = [Pair{String,String}["name" => _json(lk.sectors[i]),
             "color" => _json(_PLOT_COLORS[mod1(i, length(_PLOT_COLORS))])]
             for i in 1:length(lk.sectors)]
    groups_json = _json_array_of_objects(grows)
    ref = "[" *
        _json_obj(Pair{String,String}["axis" => _json("x"), "value" => "1.0",
            "color" => _json("#999"), "dash" => _json("4,3")]) * "," *
        _json_obj(Pair{String,String}["axis" => _json("y"), "value" => "1.0",
            "color" => _json("#999"), "dash" => _json("4,3")]) * "]"
    js = _render_scatter_js(id, data_json, groups_json;
                            ref_lines_json=ref,
                            xlabel="U_i (backward)", ylabel="U_j (forward)")
    panel = _PanelSpec(id, title, js)
    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(m::LeontiefModel; title="Leontief inverse", save_path=nothing) -> PlotOutput

Heatmap of the Leontief inverse matrix `L`.
"""
function plot_result(m::LeontiefModel; title::String="Leontief inverse",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_leontief")
    secs = m.io.sectors
    n = length(secs)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n, j in 1:n
        push!(rows, Pair{String,String}["x" => _json(secs[j]), "y" => _json(secs[i]),
                                        "v" => _json(m.L[i, j])])
    end
    data_json = _json_array_of_objects(rows)
    mx = maximum(m.L)
    # Leontief inverse is nonnegative → single-hue sequential ramp over [0, max]
    # (plotrule Color / Heatmaps; a diverging scale here has a meaningless midpoint).
    js = _render_heatmap_js(id, data_json, _json(secs), _json(secs);
                            xlabel="using sector", ylabel="supplying sector",
                            scale=:sequential, color_domain=[0.0, float(mx)])
    panel = _PanelSpec(id, title, js)
    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(eq::BFEquilibrium; title="", save_path=nothing) -> PlotOutput

Bar chart of real-sector log output changes `dlog x` from an exact B&F
counterfactual.
"""
function plot_result(eq::BFEquilibrium; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_bf_eq")
    rows = [Pair{String,String}["x" => _json(eq.sectors[i]),
                                "dlog x" => _json(eq.dlog_x[i])]
            for i in 1:length(eq.sectors)]
    data_json = _json_array_of_objects(rows)
    series_json = "[" * _json_obj(Pair{String,String}[
        "key" => _json("dlog x"), "name" => _json("dlog x"),
        "color" => _json(_PLOT_COLORS[1])]) * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Sector", ylabel="dlog x")
    ttl = isempty(title) ? "B&F equilibrium: sectoral dlog x (dlogY=$(eq.dlogY))" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(net::ProductionNetwork; title="", save_path=nothing) -> PlotOutput

Bar chart of base Domar weights on real-sector outer nodes. With wedges, plots
cost-based (`λ̃`) and revenue-based (`λ`) side by side.
"""
function plot_result(net::ProductionNetwork; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_bf_net")
    λ = [net.lambda[g] for g in net.outer_nodes]
    λr = [net.lambda_rev[g] for g in net.outer_nodes]
    secs = net.io.sectors
    has_wedges = any(m -> m > 1 + 1e-14, net.mu)
    if has_wedges
        rows = [Pair{String,String}["x" => _json(secs[i]),
                                    "Cost-based λ̃" => _json(λ[i]),
                                    "Revenue-based λ" => _json(λr[i])]
                for i in 1:length(secs)]
        data_json = _json_array_of_objects(rows)
        series_json = "[" *
            _json_obj(Pair{String,String}[
                "key" => _json("Cost-based λ̃"), "name" => _json("Cost-based λ̃"),
                "color" => _json(_PLOT_COLORS[1])]) * "," *
            _json_obj(Pair{String,String}[
                "key" => _json("Revenue-based λ"), "name" => _json("Revenue-based λ"),
                "color" => _json(_PLOT_COLORS[2])]) * "]"
    else
        rows = [Pair{String,String}["x" => _json(secs[i]), "Domar" => _json(λ[i])]
                for i in 1:length(secs)]
        data_json = _json_array_of_objects(rows)
        series_json = "[" * _json_obj(Pair{String,String}[
            "key" => _json("Domar"), "name" => _json("Domar λ̃"),
            "color" => _json(_PLOT_COLORS[1])]) * "]"
    end
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Sector", ylabel="Domar weight")
    ttl = isempty(title) ? "ProductionNetwork Domar weights ($(net.nests))" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(w::BFWedgeDecomp; title="", save_path=nothing) -> PlotOutput

Bar chart of cost-based vs revenue-based Domar weights from a wedge decomposition.
"""
function plot_result(w::BFWedgeDecomp; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_bf_wedge")
    rows = [Pair{String,String}["x" => _json(w.sectors[i]),
                                "Cost-based λ̃" => _json(w.lambda_cost[i]),
                                "Revenue-based λ" => _json(w.lambda_rev[i])]
            for i in 1:length(w.sectors)]
    data_json = _json_array_of_objects(rows)
    series_json = "[" *
        _json_obj(Pair{String,String}[
            "key" => _json("Cost-based λ̃"), "name" => _json("Cost-based λ̃"),
            "color" => _json(_PLOT_COLORS[1])]) * "," *
        _json_obj(Pair{String,String}[
            "key" => _json("Revenue-based λ"), "name" => _json("Revenue-based λ"),
            "color" => _json(_PLOT_COLORS[2])]) * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Sector", ylabel="Domar weight")
    ttl = isempty(title) ?
        "B&F 2020 Domar: tech=$(_fmt(w.technology)), AE=$(_fmt(w.allocative))" :
        title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(bf::BFLocal; title="", save_path=nothing) -> PlotOutput

Bar chart of Hulten first-order (Domar) elasticities from a generalized local
B&F approximation.
"""
function plot_result(bf::BFLocal; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_bf_local")
    rows = [Pair{String,String}["x" => _json(bf.sectors[i]),
                                "Hulten" => _json(bf.first_order[i])]
            for i in 1:length(bf.sectors)]
    data_json = _json_array_of_objects(rows)
    series_json = "[" * _json_obj(Pair{String,String}[
        "key" => _json("Hulten"), "name" => _json("Hulten λ̃"),
        "color" => _json(_PLOT_COLORS[1])]) * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Sector", ylabel="∂logY/∂logA")
    ttl = isempty(title) ? "BFLocal Hulten elasticities ($(bf.nests))" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(e::BFElasticities; title="", save_path=nothing) -> PlotOutput

Heatmap of real-sector price incidence ``∂ log p_i / ∂ log A_j``.
"""
function plot_result(e::BFElasticities; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_bf_elast")
    secs = e.sectors
    n = length(secs)
    rows = Vector{Pair{String,String}}[]
    for i in 1:n, j in 1:n
        push!(rows, Pair{String,String}["x" => _json(secs[j]), "y" => _json(secs[i]),
                                        "v" => _json(e.dlogp_dlogA[i, j])])
    end
    data_json = _json_array_of_objects(rows)
    mx = maximum(abs, e.dlogp_dlogA; init=1.0)
    js = _render_heatmap_js(id, data_json, _json(secs), _json(secs);
                            xlabel="shock sector j", ylabel="price sector i",
                            scale=:diverging, color_domain=[-float(mx), float(mx)])
    ttl = isempty(title) ? "Price incidence ∂log p / ∂log A" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(sc::BFShockCurve; title="", save_path=nothing) -> PlotOutput

Three-series line chart of exact ``Δ log Y``, Hulten, and second-order Taylor
along a one-sector productivity shock grid (the signature B&F concavity figure).
"""
function plot_result(sc::BFShockCurve; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_bf_shock")
    rows = [Pair{String,String}[
                "x" => _json(sc.shocks[i]),
                "Exact" => _json(sc.exact[i]),
                "Hulten" => _json(sc.hulten[i]),
                "Second-order" => _json(sc.second_order[i])]
            for i in 1:length(sc.shocks)]
    data_json = _json_array_of_objects(rows)
    series_json = "[" *
        _json_obj(Pair{String,String}[
            "key" => _json("Exact"), "name" => _json("Exact"),
            "color" => _json(_PLOT_COLORS[1])]) * "," *
        _json_obj(Pair{String,String}[
            "key" => _json("Hulten"), "name" => _json("Hulten"),
            "color" => _json(_PLOT_COLORS[2])]) * "," *
        _json_obj(Pair{String,String}[
            "key" => _json("Second-order"), "name" => _json("Second-order"),
            "color" => _json(_PLOT_COLORS[3])]) * "]"
    js = _render_line_js(id, data_json, series_json;
                         xlabel="Δ log A ($(sc.sector))", ylabel="Δ log Y")
    ttl = isempty(title) ? "B&F shock curve: $(sc.sector)" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
