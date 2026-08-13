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
    plot_result(m::BFMisallocation; title="", save_path=nothing) -> PlotOutput

Grouped bar of exact distance to the frontier (`Exact L`) against the
second-order Harberger term. Includes `First-order` when that term is
nonzero (`:observed`).
"""
function plot_result(m::BFMisallocation; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_bf_misalloc")
    show_fo = abs(m.first_order) > 1e-14
    row = Pair{String,String}["x" => _json("L"),
                              "Exact L" => _json(m.distance),
                              "Second-order" => _json(m.second_order)]
    show_fo && push!(row, "First-order" => _json(m.first_order))
    data_json = _json_array_of_objects([row])
    series_json = "[" *
        _json_obj(Pair{String,String}[
            "key" => _json("Exact L"), "name" => _json("Exact L"),
            "color" => _json(_PLOT_COLORS[1])]) * "," *
        _json_obj(Pair{String,String}[
            "key" => _json("Second-order"), "name" => _json("Second-order"),
            "color" => _json(_PLOT_COLORS[2])])
    if show_fo
        series_json *= "," * _json_obj(Pair{String,String}[
            "key" => _json("First-order"), "name" => _json("First-order"),
            "color" => _json(_PLOT_COLORS[3])])
    end
    series_json *= "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Misallocation", ylabel="L")
    ttl = isempty(title) ?
        "B&F 2020 Prop 5: L=$(_fmt(m.distance)), 2nd=$(_fmt(m.second_order)) ($(m.point))" :
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

"""
    plot_result(ed::ExportDecomposition; title="", save_path=nothing) -> PlotOutput

Stacked bar of KWW (2014) DVA / RDV / FVA / PDC components of gross exports.
"""
function plot_result(ed::ExportDecomposition; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_kww")
    labs = ["DVA", "RDV", "FVA", "PDC"]
    vals = [ed.dva, ed.rdv, ed.fva, ed.pdc]
    rows = [Pair{String,String}["x" => _json(labs[i]), "Value" => _json(vals[i])]
            for i in 1:4]
    data_json = _json_array_of_objects(rows)
    series_json = "[" * _json_obj(Pair{String,String}[
        "key" => _json("Value"), "name" => _json("Value"),
        "color" => _json(_PLOT_COLORS[1])]) * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Component", ylabel="Value")
    ttl = isempty(title) ?
        "KWW export decomposition ($(ed.region); GE=$(_fmt(ed.gross_exports)))" :
        title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(vs::VerticalSpecialization; title="", save_path=nothing) -> PlotOutput

Bar chart of sectoral foreign content in exports (HIY / KWW VS).
"""
function plot_result(vs::VerticalSpecialization; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_vs")
    labs = ["sector $(i)" for i in 1:length(vs.by_sector)]
    rows = [Pair{String,String}["x" => _json(labs[i]),
                                "VS" => _json(vs.by_sector[i])]
            for i in 1:length(vs.by_sector)]
    data_json = _json_array_of_objects(rows)
    series_json = "[" * _json_obj(Pair{String,String}[
        "key" => _json("VS"), "name" => _json("Foreign content"),
        "color" => _json(_PLOT_COLORS[1])]) * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Sector", ylabel="Foreign content")
    ttl = isempty(title) ?
        "Vertical specialization ($(vs.region); share=$(_fmt(vs.vs_share)))" :
        title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::SDAResult; title="", save_path=nothing) -> PlotOutput

Grouped bar of per-factor structural-decomposition effects (first element of
each effect vector when the indicator is multi-dimensional — typically sector 1
for output SDA, stressor 1 for emission SDA). For a full multi-sector view the
stacked contributions across all components of `total` are shown when
`length(total) ≤ 24`.
"""
function plot_result(r::SDAResult; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_sda")
    facs = isempty(r.factors) ? sort(collect(keys(r.effects))) : collect(r.factors)
    n = length(r.total)
    # One group per component of the indicator (sector or stressor), series = factors
    labs = ["c$i" for i in 1:n]
    rows = Vector{Pair{String,String}}[]
    for i in 1:n
        row = Pair{String,String}["x" => _json(labs[i])]
        for f in facs
            haskey(r.effects, f) || continue
            push!(row, string(f) => _json(r.effects[f][i]))
        end
        push!(rows, row)
    end
    data_json = _json_array_of_objects(rows)
    series_parts = String[]
    for (k, f) in enumerate(facs)
        haskey(r.effects, f) || continue
        push!(series_parts, _json_obj(Pair{String,String}[
            "key" => _json(string(f)), "name" => _json(string(f)),
            "color" => _json(_PLOT_COLORS[mod1(k, length(_PLOT_COLORS))])]))
    end
    series_json = "[" * join(series_parts, ",") * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="stacked", xlabel="Component", ylabel="Effect")
    on_str = r.on isa Symbol ? String(r.on) : string(r.on)
    ttl = isempty(title) ? "SDA ($(r.method); on=$(on_str))" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(r::RASResult; title="", save_path=nothing) -> PlotOutput

Heatmap of the balanced matrix ``X``.
"""
function plot_result(r::RASResult; title::String="",
                     save_path::Union{String,Nothing}=nothing)
    id = _next_plot_id("io_ras")
    m, n = size(r.X)
    row_labs = [string(i) for i in 1:m]
    col_labs = [string(j) for j in 1:n]
    rows = [Pair{String,String}[
                "x" => _json(col_labs[j]), "y" => _json(row_labs[i]),
                "v" => _json(r.X[i, j])]
            for i in 1:m for j in 1:n]
    data_json = _json_array_of_objects(rows)
    mx = maximum(abs, r.X; init=0.0)
    mx = mx > 0 ? mx : 1.0
    js = _render_heatmap_js(id, data_json, _json(row_labs), _json(col_labs);
                            xlabel="Column", ylabel="Row",
                            scale=:diverging, color_domain=[-float(mx), float(mx)])
    ttl = isempty(title) ?
        "RAS/GRAS balanced matrix ($(r.method); iters=$(r.iterations))" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end

"""
    plot_result(fp::RegionalFootprintResult; title="", save_path=nothing) -> PlotOutput

Grouped bar of production-based vs consumption-based totals by region (first
stressor row).
"""
function plot_result(fp::RegionalFootprintResult; title::String="",
                     save_path::Union{String,Nothing}=nothing)

    id = _next_plot_id("io_rfp")
    s = 1                                          # first stressor
    rows = [Pair{String,String}[
                "x" => _json(fp.regions[r]),
                "Production" => _json(fp.production[s, r]),
                "Consumption" => _json(fp.consumption[s, r])]
            for r in 1:length(fp.regions)]
    data_json = _json_array_of_objects(rows)
    series_json = "[" *
        _json_obj(Pair{String,String}[
            "key" => _json("Production"), "name" => _json("Production"),
            "color" => _json(_PLOT_COLORS[1])]) * "," *
        _json_obj(Pair{String,String}[
            "key" => _json("Consumption"), "name" => _json("Consumption"),
            "color" => _json(_PLOT_COLORS[2])]) * "]"
    js = _render_bar_js(id, data_json, series_json;
                        mode="grouped", xlabel="Region", ylabel=fp.stressors[s])
    ttl = isempty(title) ?
        "Regional footprint ($(fp.name) / $(fp.stressors[s]))" : title
    panel = _PanelSpec(id, ttl, js)
    p = _make_plot([panel]; title=ttl, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
