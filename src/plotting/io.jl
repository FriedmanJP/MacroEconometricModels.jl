# io.jl — plotting recipes for Input-Output results (inline D3.js)

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
    js = _render_heatmap_js(id, data_json, _json(secs), _json(secs);
                            xlabel="using sector", ylabel="supplying sector",
                            color_domain=[0.0, float(mx)])
    panel = _PanelSpec(id, title, js)
    p = _make_plot([panel]; title=title, ncols=1)
    save_path !== nothing && save_plot(p, save_path)
    p
end
