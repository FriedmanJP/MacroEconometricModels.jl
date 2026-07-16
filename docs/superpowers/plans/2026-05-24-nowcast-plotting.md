# Nowcast Plotting Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 new nowcast plot views (group news, sorted individual, contributions, heatmap, DFM factors) and supporting struct/renderer changes.

**Architecture:** Extend existing `plot_result` functions with a `view` keyword for dispatch. Add `group_names` field to `NowcastNews` struct. Add `_render_heatmap_js` D3.js renderer to `render.jl`. All new views follow the existing pattern: build JSON data → call renderer → wrap in `_PanelSpec` → compose with `_make_plot`.

**Tech Stack:** Julia, D3.js v7, inline HTML rendering

---

### Task 1: Add `group_names` field to `NowcastNews` struct

**Files:**
- Modify: `src/nowcast/types.jl:195-203`
- Modify: `src/nowcast/news.jl:45-135`
- Modify: `src/summary_nowcast.jl:96-130`
- Test: `test/nowcast/test_nowcast.jl`

- [ ] **Step 1: Write failing tests for `group_names` field**

Add these tests at the end of the "Group impacts" testset in `test/nowcast/test_nowcast.jl` (after line 620, inside the existing `@testset "Group impacts"` block):

```julia
        # group_names auto-generated
        @test length(news.group_names) == 3
        @test news.group_names[1] == "Group 1"
        @test news.group_names[3] == "Group 3"

        # group_names explicit
        news2 = nowcast_news(Y, X_old, m, 58; target_var=5, groups=groups,
                             group_names=["Ind. Prod.", "Retail", "GDP"])
        @test news2.group_names == ["Ind. Prod.", "Retail", "GDP"]
```

Also add a new testset after the "Group impacts" testset, before the closing `end` of the "News Decomposition" section:

```julia
    @testset "Default group_names without groups" begin
        Y, _, _, _ = _make_nowcast_data(T_obs=60, nM=4, nQ=1, r=1, seed=1150)
        m = nowcast_dfm(Y, 4, 1; r=1, p=1, max_iter=20, thresh=1e-3)

        X_old = copy(Y)
        X_old[58:60, 1:2] .= NaN

        news = nowcast_news(Y, X_old, m, 58; target_var=5)
        @test length(news.group_names) == 5  # one per variable
        @test news.group_names[1] == "Var1"
        @test news.group_names[5] == "Var5"
    end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()' -- nowcast`

Expected: Errors about `group_names` field not existing on `NowcastNews`.

- [ ] **Step 3: Add `group_names` field to `NowcastNews` struct**

In `src/nowcast/types.jl`, replace lines 195-203:

```julia
struct NowcastNews{T<:AbstractFloat}
    old_nowcast::T
    new_nowcast::T
    impact_news::Vector{T}
    impact_revision::T
    impact_reestimation::T
    group_impacts::Vector{T}
    variable_names::Vector{String}
end
```

with:

```julia
struct NowcastNews{T<:AbstractFloat}
    old_nowcast::T
    new_nowcast::T
    impact_news::Vector{T}
    impact_revision::T
    impact_reestimation::T
    group_impacts::Vector{T}
    group_names::Vector{String}
    variable_names::Vector{String}
end
```

Also update the docstring field list at lines 188-193, adding between `group_impacts` and `variable_names`:

```
- `group_names::Vector{String}` — labels for each group in group_impacts
```

- [ ] **Step 4: Update `nowcast_news()` to populate `group_names`**

In `src/nowcast/news.jl`, update the function signature at line 45-48:

```julia
function nowcast_news(X_new::AbstractMatrix, X_old::AbstractMatrix,
                      model::NowcastDFM{T}, target_period::Int;
                      target_var::Int=size(X_new, 2),
                      groups::Union{Vector{Int},Nothing}=nothing,
                      group_names::Union{Vector{String},Nothing}=nothing) where {T<:AbstractFloat}
```

Update the docstring at line 36 to add the new keyword:

```
- `group_names::Union{Vector{String},Nothing}` — labels for each group (auto-generated if omitted)
```

Replace the group aggregation block (lines 113-131) with:

```julia
    # Group aggregation
    if groups !== nothing
        n_groups = maximum(groups)
        group_impacts = zeros(T, n_groups)
        for (k, idx) in enumerate(i_new)
            v_k = idx[2]
            if v_k <= length(groups)
                g = groups[v_k]
                group_impacts[g] += impact_news[k]
            end
        end
        gn = if group_names !== nothing
            group_names
        else
            ["Group $i" for i in 1:n_groups]
        end
    else
        # Default: one group per variable
        group_impacts = zeros(T, N)
        for (k, idx) in enumerate(i_new)
            v_k = idx[2]
            group_impacts[v_k] += impact_news[k]
        end
        gn = ["Var$i" for i in 1:N]
    end
```

Replace the constructor call at lines 133-134:

```julia
    NowcastNews{T}(now_old, now_new, impact_news, impact_revision,
                   impact_reestimation, group_impacts, gn, variable_names)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && MACRO_MULTIPROCESS_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()' -- nowcast`

Expected: All nowcast tests pass, including the new group_names tests.

- [ ] **Step 6: Commit**

```bash
git add src/nowcast/types.jl src/nowcast/news.jl test/nowcast/test_nowcast.jl
git commit -m "feat(nowcast): add group_names field to NowcastNews struct"
```

---

### Task 2: Add `_render_heatmap_js` D3.js renderer

**Files:**
- Modify: `src/plotting/render.jl:444` (insert before "Panel Body Rendering" section)
- Test: `test/plotting/test_plot_result.jl`

- [ ] **Step 1: Write failing test for heatmap renderer**

Add this testset inside the main `@testset "Plotting — plot_result()"` block in `test/plotting/test_plot_result.jl`, after the existing "NowcastNews" testset (after line 357) and before the "Infrastructure" testset:

```julia
    @testset "NowcastResult heatmap view" begin
        nM = 4; nQ = 1
        Y_nc = randn(Random.MersenneTwister(77), 100, nM + nQ)
        Y_nc[end, end] = NaN
        Y_nc[end-1:end, 3] .= NaN
        dfm_nc = nowcast_dfm(Y_nc, nM, nQ; r=2, p=1)
        nr = nowcast(dfm_nc)
        p = plot_result(nr; view=:heatmap)
        check_plot(p)
        @test occursin("heatmap", lowercase(p.html)) || occursin("interpolateRdBu", p.html)
        @test occursin("#d9d9d9", p.html)  # grey for NaN cells
    end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels, Test, Random; include("test/plotting/test_plot_result.jl")'`

Expected: FAIL — `view` keyword not recognized or `_render_heatmap_js` not defined.

- [ ] **Step 3: Add `_render_heatmap_js` to `render.jl`**

Insert the following block in `src/plotting/render.jl` before the "Panel Body Rendering" comment (before line 446):

```julia
# =============================================================================
# Heatmap Chart Renderer
# =============================================================================

"""
Generate D3.js code for a color-coded heatmap matrix.
- `data_json`: JSON array of {x, y, v} objects (v can be null for missing)
- `row_labels_json`: JSON array of row label strings
- `col_labels_json`: JSON array of column label strings
- `color_domain`: [min, max] for the diverging color scale
"""
function _render_heatmap_js(id::String, data_json::String,
                            row_labels_json::String, col_labels_json::String;
                            xlabel::String="", ylabel::String="",
                            color_domain::Vector{<:Real}=[-3, 3])
    cmin, cmax = color_domain[1], color_domain[2]
    """
(function() {
    const data = $(data_json);
    const rowLabels = $(row_labels_json);
    const colLabels = $(col_labels_json);

    const container = d3.select('#$(id)');
    const W = Math.max(container.node().clientWidth - 24, 280);
    const maxLabelW = 120;
    const margin = {top:10, right:15, bottom:35, left: maxLabelW};
    const w = W - margin.left - margin.right;
    const cellH = Math.max(Math.min(18, 250 / rowLabels.length), 8);
    const h = cellH * rowLabels.length;

    const svg = container.append('svg').attr('width', W).attr('height', h + margin.top + margin.bottom);
    const g = svg.append('g').attr('transform', 'translate('+margin.left+','+margin.top+')');

    const x = d3.scaleBand().domain(colLabels).range([0, w]).padding(0.02);
    const y = d3.scaleBand().domain(rowLabels).range([0, h]).padding(0.02);

    const color = d3.scaleSequential(t => d3.interpolateRdBu(1 - t))
        .domain([$(cmin), $(cmax)]);
    const grey = '#d9d9d9';

    g.selectAll('rect.cell').data(data).join('rect').attr('class','cell')
        .attr('x', d => x(d.x))
        .attr('y', d => y(d.y))
        .attr('width', x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', d => d.v === null ? grey : color(Math.max($(cmin), Math.min($(cmax), d.v))))
        .attr('rx', 1)
        .on('mouseover', function(evt, d) {
            const val = d.v === null ? 'Missing' : fmt(d.v);
            showTip(evt, '<b>'+d.y+'</b><br>Period '+d.x+': '+val);
        })
        .on('mouseout', hideTip);

    // Axes
    g.append('g').attr('class','axis').attr('transform','translate(0,'+h+')')
        .call(d3.axisBottom(x).tickValues(colLabels.filter((d,i) =>
            i % Math.max(1, Math.floor(colLabels.length/10)) === 0)));
    g.append('g').attr('class','axis')
        .call(d3.axisLeft(y).tickFormat(d => d.length > 20 ? d.slice(0,18)+'..' : d));

    if('$(xlabel)') g.append('text').attr('x',w/2).attr('y',h+30).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(xlabel)');
    if('$(ylabel)') g.append('text').attr('transform','rotate(-90)')
        .attr('x',-h/2).attr('y',-maxLabelW+10).attr('text-anchor','middle')
        .attr('font-size','11px').attr('fill','#666').text('$(ylabel)');
})();
"""
end

```

- [ ] **Step 4: Commit**

```bash
git add src/plotting/render.jl
git commit -m "feat(plotting): add _render_heatmap_js D3.js renderer"
```

---

### Task 3: Rewrite `plot_result(::NowcastNews)` with `view` dispatch

**Files:**
- Modify: `src/plotting/nowcast.jl:86-123`
- Test: `test/plotting/test_plot_result.jl`

- [ ] **Step 1: Write failing tests for new NowcastNews views**

Add these testsets in `test/plotting/test_plot_result.jl`, after the existing "NowcastNews" testset (after line 357):

```julia
    @testset "NowcastNews view=:groups" begin
        X_old = randn(Random.MersenneTwister(55), 100, 5)
        X_old[end, end] = NaN
        X_new = copy(X_old)
        X_new[end, end] = 0.5
        dfm2 = nowcast_dfm(X_old, 4, 1; r=2, p=1)
        groups = [1, 1, 2, 2, 3]
        nn = nowcast_news(X_new, X_old, dfm2, 5; groups=groups,
                          group_names=["Industry", "Retail", "GDP"])
        p = plot_result(nn; view=:groups)
        check_plot(p)
        @test occursin("Industry", p.html)
        @test occursin("Retail", p.html)
    end

    @testset "NowcastNews view=:individual" begin
        X_old = randn(Random.MersenneTwister(56), 100, 5)
        X_old[98:100, 1:2] .= NaN
        X_new = copy(X_old)
        X_new[98:100, 1:2] .= randn(Random.MersenneTwister(57), 3, 2)
        dfm2 = nowcast_dfm(X_old, 4, 1; r=2, p=1)
        nn = nowcast_news(X_new, X_old, dfm2, 5)
        p = plot_result(nn; view=:individual)
        check_plot(p)
        @test occursin("Impact", p.html)
    end

    @testset "NowcastNews invalid view" begin
        X_old = randn(Random.MersenneTwister(58), 100, 5)
        X_old[end, end] = NaN
        X_new = copy(X_old)
        X_new[end, end] = 0.5
        dfm2 = nowcast_dfm(X_old, 4, 1; r=2, p=1)
        nn = nowcast_news(X_new, X_old, dfm2, 5)
        @test_throws ArgumentError plot_result(nn; view=:nonexistent)
    end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels, Test, Random; include("test/plotting/test_plot_result.jl")'`

Expected: FAIL — `view` keyword not accepted by current `plot_result(::NowcastNews)`.

- [ ] **Step 3: Rewrite `plot_result(::NowcastNews)` with view dispatch**

Replace lines 86-123 in `src/plotting/nowcast.jl` with:

```julia
# =============================================================================
# NowcastNews
# =============================================================================

"""
    plot_result(nn::NowcastNews; view=:releases, title="", save_path=nothing)

Plot nowcast news decomposition.

# Views
- `:releases` — horizontal bar chart of per-release impact (default)
- `:groups` — stacked bar chart of group impacts
- `:individual` — horizontal bar chart sorted by absolute impact
"""
function plot_result(nn::NowcastNews{T};
                     view::Symbol=:releases,
                     title::String="",
                     save_path::Union{String,Nothing}=nothing) where {T}
    if view == :releases
        p = _plot_news_releases(nn; title=title)
    elseif view == :groups
        p = _plot_news_groups(nn; title=title)
    elseif view == :individual
        p = _plot_news_individual(nn; title=title)
    else
        throw(ArgumentError("Unknown view: $view. Expected :releases, :groups, or :individual"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _news_title(nn::NowcastNews{T}, prefix::String) where {T}
    delta = nn.new_nowcast - nn.old_nowcast
    isempty(prefix) ? "Nowcast News: $(round(nn.old_nowcast, digits=3)) → $(round(nn.new_nowcast, digits=3)) (Δ = $(round(delta, digits=3)))" : prefix
end

function _plot_news_releases(nn::NowcastNews{T}; title::String="") where {T}
    id = _next_plot_id("news")
    n_releases = length(nn.impact_news)

    rows = Vector{Pair{String,String}}[]
    for i in 1:n_releases
        label = i <= length(nn.variable_names) ? nn.variable_names[i] : "Release $i"
        push!(rows, [
            "x" => _json(label),
            "impact" => _json(nn.impact_news[i])
        ])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["News Impact"], [_PLOT_COLORS[1]]; keys=["impact"])

    js = _render_bar_js(id, data_json, s_json; mode="grouped",
                        xlabel="", ylabel="Impact")

    _make_plot([_PanelSpec(id, "Per-Release Impact", js)];
               title=_news_title(nn, title))
end

function _plot_news_groups(nn::NowcastNews{T}; title::String="") where {T}
    id = _next_plot_id("news_grp")
    n_groups = length(nn.group_impacts)

    # Build stacked bar data: one bar "News", optionally "Revision + Re-est."
    bar_labels = String["News"]
    has_other = abs(nn.impact_revision) + abs(nn.impact_reestimation) > T(1e-10)
    if has_other
        push!(bar_labels, "Revision + Re-est.")
    end

    rows = Vector{Pair{String,String}}[]
    # News bar: one series per group
    row_news = Pair{String,String}["x" => _json("News")]
    for g in 1:n_groups
        push!(row_news, "g$g" => _json(nn.group_impacts[g]))
    end
    push!(rows, row_news)

    if has_other
        row_other = Pair{String,String}["x" => _json("Revision + Re-est.")]
        for g in 1:n_groups
            push!(row_other, "g$g" => _json(g == 1 ? nn.impact_revision + nn.impact_reestimation : zero(T)))
        end
        push!(rows, row_other)
    end

    data_json = _json_array_of_objects(rows)

    names = String[]
    colors = String[]
    keys_arr = String[]
    for g in 1:n_groups
        push!(names, g <= length(nn.group_names) ? nn.group_names[g] : "Group $g")
        push!(colors, _PLOT_COLORS[mod1(g, length(_PLOT_COLORS))])
        push!(keys_arr, "g$g")
    end
    s_json = _series_json(names, colors; keys=keys_arr)

    js = _render_bar_js(id, data_json, s_json; mode="stacked",
                        xlabel="", ylabel="Impact")

    _make_plot([_PanelSpec(id, "News by Group", js)];
               title=_news_title(nn, title))
end

function _plot_news_individual(nn::NowcastNews{T}; title::String="") where {T}
    id = _next_plot_id("news_ind")
    n_releases = length(nn.impact_news)

    # Sort by absolute impact descending
    sorted_idx = sortperm(abs.(nn.impact_news), rev=true)

    rows = Vector{Pair{String,String}}[]
    for i in sorted_idx
        label = i <= length(nn.variable_names) ? nn.variable_names[i] : "Release $i"
        push!(rows, [
            "x" => _json(label),
            "impact" => _json(nn.impact_news[i])
        ])
    end
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["News Impact"], [_PLOT_COLORS[1]]; keys=["impact"])

    js = _render_bar_js(id, data_json, s_json; mode="grouped",
                        xlabel="", ylabel="Impact")

    _make_plot([_PanelSpec(id, "Individual Impacts (sorted)", js)];
               title=_news_title(nn, title))
end
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels, Test, Random; include("test/plotting/test_plot_result.jl")'`

Expected: All NowcastNews tests pass (releases, groups, individual, invalid view).

- [ ] **Step 5: Commit**

```bash
git add src/plotting/nowcast.jl test/plotting/test_plot_result.jl
git commit -m "feat(plotting): add view dispatch to plot_result(::NowcastNews)"
```

---

### Task 4: Rewrite `plot_result(::NowcastResult)` with `view` dispatch

**Files:**
- Modify: `src/plotting/nowcast.jl:14-84`
- Test: `test/plotting/test_plot_result.jl`

- [ ] **Step 1: Write failing tests for new NowcastResult views**

Add these testsets in `test/plotting/test_plot_result.jl` after the existing "NowcastResult" testset (after line 345, before the NowcastNews testsets):

```julia
    @testset "NowcastResult view=:default with DFM factors" begin
        nM = 4; nQ = 1
        Y_nc = randn(Random.MersenneTwister(70), 100, nM + nQ)
        Y_nc[end, end] = NaN
        dfm_nc = nowcast_dfm(Y_nc, nM, nQ; r=2, p=1)
        nr = nowcast(dfm_nc)
        p = plot_result(nr; view=:default)
        check_plot(p)
        @test occursin("Factor 1", p.html)
        @test occursin("Factor 2", p.html)
    end

    @testset "NowcastResult view=:contributions" begin
        nM = 4; nQ = 1
        Y_nc = randn(Random.MersenneTwister(71), 100, nM + nQ)
        Y_nc[end, end] = NaN
        dfm_nc = nowcast_dfm(Y_nc, nM, nQ; r=2, p=1)
        nr = nowcast(dfm_nc)
        p = plot_result(nr; view=:contributions)
        check_plot(p)
        @test occursin("Contribution", p.html) || occursin("contribution", p.html)
    end

    @testset "NowcastResult view=:contributions requires DFM" begin
        rng = Random.MersenneTwister(72)
        Y = randn(rng, 60, 4)
        Y[55:60, 3:4] .= NaN
        m = nowcast_bvar(Y, 2, 2; lags=2, max_iter=20)
        nr = nowcast(m)
        @test_throws ArgumentError plot_result(nr; view=:contributions)
    end

    @testset "NowcastResult invalid view" begin
        nM = 4; nQ = 1
        Y_nc = randn(Random.MersenneTwister(73), 100, nM + nQ)
        Y_nc[end, end] = NaN
        dfm_nc = nowcast_dfm(Y_nc, nM, nQ; r=2, p=1)
        nr = nowcast(dfm_nc)
        @test_throws ArgumentError plot_result(nr; view=:bad)
    end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels, Test, Random; include("test/plotting/test_plot_result.jl")'`

Expected: FAIL — `view` keyword not accepted, `:contributions` not implemented.

- [ ] **Step 3: Rewrite `plot_result(::NowcastResult)` with view dispatch**

Replace lines 14-84 in `src/plotting/nowcast.jl` with:

```julia
# =============================================================================
# NowcastResult
# =============================================================================

"""
    plot_result(nr::NowcastResult; view=:default, ncols=0, title="", save_path=nothing, kwargs...)

Plot nowcast result.

# Views
- `:default` — smoothed data with nowcast/forecast extension (+ DFM factor panels)
- `:heatmap` — z-score heatmap of input variables with ragged edge
- `:contributions` — group contributions stacked bar (DFM only)

# Keyword Arguments for `:heatmap` and `:contributions`
- `groups::Union{Vector{Int},Nothing}` — variable-to-group mapping
- `group_names::Union{Vector{String},Nothing}` — labels for groups
- `variable_names::Union{Vector{String},Nothing}` — labels for variables
- `n_periods::Int=18` — number of trailing periods for heatmap
"""
function plot_result(nr::NowcastResult{T};
                     view::Symbol=:default,
                     ncols::Int=0, title::String="",
                     save_path::Union{String,Nothing}=nothing,
                     groups::Union{Vector{Int},Nothing}=nothing,
                     group_names::Union{Vector{String},Nothing}=nothing,
                     variable_names::Union{Vector{String},Nothing}=nothing,
                     n_periods::Int=18) where {T}
    if view == :default
        p = _plot_nowcast_default(nr; ncols=ncols, title=title)
    elseif view == :heatmap
        p = _plot_nowcast_heatmap(nr; title=title, groups=groups,
                                  group_names=group_names,
                                  variable_names=variable_names,
                                  n_periods=n_periods)
    elseif view == :contributions
        p = _plot_nowcast_contributions(nr; title=title, ncols=ncols,
                                        groups=groups, group_names=group_names)
    else
        throw(ArgumentError("Unknown view: $view. Expected :default, :heatmap, or :contributions"))
    end
    save_path !== nothing && save_plot(p, save_path)
    p
end

function _plot_nowcast_default(nr::NowcastResult{T};
                               ncols::Int=0, title::String="") where {T}
    T_obs, n_vars = size(nr.X_sm)
    n_show = min(n_vars, 6)

    panels = _PanelSpec[]

    # Target variable panel with nowcast/forecast extension
    ti = nr.target_index
    id = _next_plot_id("nc_target")
    ptitle = "Target (col $ti) — Nowcast: $(round(nr.nowcast, digits=3)), Forecast: $(round(nr.forecast, digits=3))"

    rows = Vector{Pair{String,String}}[]
    for t in 1:T_obs
        push!(rows, [
            "x" => _json(t),
            "v1" => _json(nr.X_sm[t, ti]),
            "v2" => _json(t == T_obs ? nr.X_sm[t, ti] : NaN)
        ])
    end
    push!(rows, [
        "x" => _json(T_obs + 1),
        "v1" => "null",
        "v2" => _json(nr.nowcast)
    ])
    push!(rows, [
        "x" => _json(T_obs + 2),
        "v1" => "null",
        "v2" => _json(nr.forecast)
    ])
    data_json = _json_array_of_objects(rows)
    s_json = _series_json(["Smoothed", "Nowcast/Forecast"],
                          [_PLOT_COLORS[1], _PLOT_COLORS[2]]; keys=["v1", "v2"])
    js = _render_line_js(id, data_json, s_json; xlabel="Period", ylabel="Value")
    push!(panels, _PanelSpec(id, ptitle, js))

    # Additional variable panels
    other_vars = setdiff(1:min(n_vars, n_show + 1), [ti])
    for vi in other_vars[1:min(length(other_vars), n_show - 1)]
        id_v = _next_plot_id("nc_var")

        rows_v = Vector{Pair{String,String}}[]
        for t in 1:T_obs
            push!(rows_v, [
                "x" => _json(t),
                "v1" => _json(nr.X_sm[t, vi])
            ])
        end
        data_v = _json_array_of_objects(rows_v)
        s_v = _series_json(["Smoothed var $vi"], [_PLOT_COLORS[mod1(vi, length(_PLOT_COLORS))]];
                           keys=["v1"])
        js_v = _render_line_js(id_v, data_v, s_v; xlabel="Period", ylabel="")
        push!(panels, _PanelSpec(id_v, "Variable $vi", js_v))
    end

    # DFM factor panels
    if nr.model isa NowcastDFM
        dfm = nr.model
        r = dfm.r
        for fi in 1:r
            id_f = _next_plot_id("nc_factor")
            rows_f = Vector{Pair{String,String}}[]
            for t in 1:T_obs
                push!(rows_f, [
                    "x" => _json(t),
                    "v1" => _json(dfm.F[t, fi])
                ])
            end
            data_f = _json_array_of_objects(rows_f)
            s_f = _series_json(["Factor $fi"], [_PLOT_COLORS[mod1(fi + n_show, length(_PLOT_COLORS))]];
                               keys=["v1"])
            js_f = _render_line_js(id_f, data_f, s_f; xlabel="Period", ylabel="")
            push!(panels, _PanelSpec(id_f, "Factor $fi", js_f))
        end
    end

    if isempty(title)
        title = "Nowcast Result ($(nr.method))"
    end

    _make_plot(panels; title=title, ncols=ncols)
end

function _plot_nowcast_heatmap(nr::NowcastResult{T};
                               title::String="",
                               groups::Union{Vector{Int},Nothing}=nothing,
                               group_names::Union{Vector{String},Nothing}=nothing,
                               variable_names::Union{Vector{String},Nothing}=nothing,
                               n_periods::Int=18) where {T}
    data = nr.model.data
    T_obs, N = size(data)

    # Variable names
    vnames = if variable_names !== nothing
        variable_names
    else
        ["Var $i" for i in 1:N]
    end

    # Compute z-scores per column (using non-NaN values)
    z_scores = Matrix{Union{T,Nothing}}(nothing, T_obs, N)
    for j in 1:N
        col = data[:, j]
        valid = .!isnan.(col)
        if count(valid) > 1
            mu = mean(col[valid])
            sd = std(col[valid])
            if sd > T(1e-10)
                for t in 1:T_obs
                    if valid[t]
                        z_scores[t, j] = (col[t] - mu) / sd
                    end
                end
            end
        end
    end

    # Select last n_periods
    t_start = max(1, T_obs - n_periods + 1)
    t_end = T_obs

    # Row ordering by groups
    if groups !== nothing
        row_order = sortperm(groups)
    else
        row_order = collect(1:N)
    end

    # Build heatmap data
    col_labels = [string(t) for t in t_start:t_end]
    row_labels = [vnames[i] for i in row_order]

    rows = Vector{Pair{String,String}}[]
    for vi in row_order
        for t in t_start:t_end
            val = z_scores[t, vi]
            push!(rows, [
                "x" => _json(string(t)),
                "y" => _json(vnames[vi]),
                "v" => val === nothing ? "null" : _json(val)
            ])
        end
    end
    data_json = _json_array_of_objects(rows)

    id = _next_plot_id("nc_heat")
    js = _render_heatmap_js(id, data_json, _json(row_labels), _json(col_labels);
                            xlabel="Period", ylabel="")

    if isempty(title)
        title = "Input Variable Heatmap (z-scores)"
    end

    _make_plot([_PanelSpec(id, "Z-Score Heatmap", js)]; title=title)
end

function _plot_nowcast_contributions(nr::NowcastResult{T};
                                     title::String="",
                                     ncols::Int=0,
                                     groups::Union{Vector{Int},Nothing}=nothing,
                                     group_names::Union{Vector{String},Nothing}=nothing) where {T}
    if !(nr.model isa NowcastDFM)
        throw(ArgumentError("view=:contributions requires NowcastDFM model (got $(typeof(nr.model)))"))
    end

    dfm = nr.model
    ti = nr.target_index
    r = dfm.r
    n_blocks = size(dfm.blocks, 2)

    # Determine groups from blocks if not provided
    grp = if groups !== nothing
        groups
    else
        # One group per block column: assign each factor to the block that loads it
        block_groups = zeros(Int, r)
        for bi in 1:n_blocks
            for fi in 1:r
                if fi <= size(dfm.blocks, 1) && dfm.blocks[fi, bi] == 1
                    block_groups[fi] = bi
                end
            end
        end
        # Fallback: unassigned factors go to group n_blocks+1
        for fi in 1:r
            if block_groups[fi] == 0
                block_groups[fi] = n_blocks + 1
            end
        end
        block_groups
    end

    n_groups = maximum(grp)
    gnames = if group_names !== nothing
        group_names
    else
        ["Block $i" for i in 1:n_groups]
    end

    # Compute group contributions: C[target, factor_cols] * F[t, factor_cols]
    # For nowcast (last period) and forecast (extrapolated one step via A)
    T_obs = size(dfm.F, 1)
    state_dim = size(dfm.A, 1)

    contrib_now = zeros(T, n_groups)
    for fi in 1:r
        g = fi <= length(grp) ? grp[fi] : n_groups
        contrib_now[g] += dfm.C[ti, fi] * dfm.F[T_obs, fi]
    end

    # Forecast: F_{T+1} = A * F_T
    F_forecast = dfm.A * dfm.F[T_obs, 1:state_dim]
    contrib_fcast = zeros(T, n_groups)
    for fi in 1:r
        g = fi <= length(grp) ? grp[fi] : n_groups
        contrib_fcast[g] += dfm.C[ti, fi] * F_forecast[fi]
    end

    # Mean contribution
    mean_val = dfm.Mx[ti]

    # Build stacked bar data
    rows = Vector{Pair{String,String}}[]

    row_now = Pair{String,String}["x" => _json("Nowcast")]
    for g in 1:n_groups
        push!(row_now, "g$g" => _json(contrib_now[g] * dfm.Wx[ti]))
    end
    push!(row_now, "mean" => _json(mean_val))
    push!(rows, row_now)

    row_fc = Pair{String,String}["x" => _json("Forecast")]
    for g in 1:n_groups
        push!(row_fc, "g$g" => _json(contrib_fcast[g] * dfm.Wx[ti]))
    end
    push!(row_fc, "mean" => _json(mean_val))
    push!(rows, row_fc)

    data_json = _json_array_of_objects(rows)

    names = String[]
    colors = String[]
    keys_arr = String[]
    for g in 1:n_groups
        push!(names, g <= length(gnames) ? gnames[g] : "Block $g")
        push!(colors, _PLOT_COLORS[mod1(g, length(_PLOT_COLORS))])
        push!(keys_arr, "g$g")
    end
    push!(names, "Mean")
    push!(colors, _PLOT_COLORS[mod1(n_groups + 1, length(_PLOT_COLORS))])
    push!(keys_arr, "mean")
    s_json = _series_json(names, colors; keys=keys_arr)

    id = _next_plot_id("nc_contrib")
    js = _render_bar_js(id, data_json, s_json; mode="stacked",
                        xlabel="", ylabel="Contribution")

    if isempty(title)
        title = "Group Contributions to Nowcast"
    end

    _make_plot([_PanelSpec(id, "Contributions by Group", js)]; title=title, ncols=ncols)
end
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels, Test, Random; include("test/plotting/test_plot_result.jl")'`

Expected: All NowcastResult tests pass (default with factors, heatmap, contributions, contributions-requires-DFM, invalid view).

- [ ] **Step 5: Commit**

```bash
git add src/plotting/nowcast.jl test/plotting/test_plot_result.jl
git commit -m "feat(plotting): add view dispatch to plot_result(::NowcastResult)"
```

---

### Task 5: Run full plotting test suite and fix any issues

**Files:**
- Test: `test/plotting/test_plot_result.jl`

- [ ] **Step 1: Run the full plotting test file**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels, Test, Random, DataFrames; include("test/plotting/test_plot_result.jl")'`

Expected: ALL tests pass — existing plots (IRF, FEVD, HD, filters, forecast, volatility, spectral, etc.) are unaffected.

- [ ] **Step 2: Run the nowcast test suite to verify struct change doesn't break anything**

Run: `cd /Users/chung/Desktop/CODES/MacroEconometricModels && julia --project=. -e 'using MacroEconometricModels, Test, Random, DataFrames; include("test/nowcast/test_nowcast.jl")'`

Expected: All nowcast tests pass, including the new group_names tests from Task 1.

- [ ] **Step 3: Fix any failures**

If any test fails, diagnose and fix. Common issues to watch for:
- `NowcastNews` constructor arity mismatch in test helpers
- `NowcastDFM.F` matrix dimensions (`T_obs × state_dim` where `state_dim = r * p`, not `T_obs × r`) — factor panel loop must use `dfm.F[:, fi]` where `fi` ranges `1:r`
- `dfm.A * dfm.F[T_obs, 1:state_dim]` dimension mismatch if `state_dim != size(A, 1)`

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix(plotting): resolve test failures in nowcast plot views"
```

(Skip this step if no fixes were needed.)
