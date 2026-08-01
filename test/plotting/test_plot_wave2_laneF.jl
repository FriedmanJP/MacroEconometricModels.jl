# =============================================================================
# Lane F — HA-DSGE dynamics + OLG/continuous-time plots (PLT-34, PLT-35).
#
# Covers plotrule Testing Rules 1–7 for every new dispatch:
#   PLT-34  plot_result(::HADSGESolution; view=:distribution_dynamics/:inequality)
#           plot_result(::KrusellSmithSolution)   plot_result(::DenHaanAccuracy)
#   PLT-35  plot_result(::CTSteadyState; view=:distribution/:policy/:lorenz)
#           plot_result(::CTTransition)  plot_result(::CTTwoAssetSolution)
#           plot_result(::BlanchardOLGSolution)
#
# Structural assertions parse the EXTRACTED JSON literals (extract_json_blocks /
# assert_strict_json / assert_all_json_valid), never raw p.html (which inlines the
# vendored D3 blob whose minified source contains NaN/tickValues tokens).
# =============================================================================

using Test, Random, SparseArrays
using MacroEconometricModels
const _MF = MacroEconometricModels

# Shared assertions (self-bootstrap when this file is run standalone).
isdefined(Main, :check_plot) || include(joinpath(@__DIR__, "plot_test_helpers.jl"))

# -----------------------------------------------------------------------------
# Synthetic fixtures (fast; plot methods only read the fields they draw).
# -----------------------------------------------------------------------------

function _laneF_ct_ss(; I::Int=25, seed::Int=11, nan_c::Bool=false)
    rng = MersenneTwister(seed)
    a = collect(range(0.0, 12.0; length=I))
    g = abs.(randn(rng, I, 2)) .+ 0.1
    da = a[2] - a[1]
    g ./= (sum(g) * da)                                   # ∫ g da = 1
    c = repeat(0.5 .+ 0.3 .* a, 1, 2); c[:, 2] .+= 0.2
    nan_c && (c[3, 1] = NaN)
    s = repeat(0.1 .* (6 .- a), 1, 2)
    v = -abs.(randn(rng, I, 2))
    _MF.CTSteadyState{Float64}(0.03, 1.2, 4.0, 1.0, a, g, v, c, s, spzeros(2I, 2I), true)
end

function _laneF_ct_transition(; N::Int=30, nan_K::Bool=false)
    t = collect(range(0.0, 20.0; length=N))
    K = 4.0 .- 0.5 .* exp.(-t)
    nan_K && (K[5] = NaN)
    _MF.CTTransition{Float64}(t, 1.0 .+ 0.1 .* exp.(-t), K,
        0.03 .+ 0.01 .* exp.(-t), 1.2 .- 0.05 .* exp.(-t), 0.9 .+ 0.02 .* exp.(-t), true, 12)
end

function _laneF_ct_two_asset(; Ib::Int=6, Ia::Int=7, nan_g::Bool=false)
    bg = collect(range(0.0, 10.0; length=Ib))
    ag = collect(range(0.0, 15.0; length=Ia))
    mk(f) = reshape([f(i, j, k) for i in 1:Ib for j in 1:Ia for k in 1:2], Ib, Ia, 2)
    V = mk((i, j, k) -> -abs(sin(i + j + k)))
    c = mk((i, j, k) -> abs(sin(i + j + k)) + 0.1)
    d = mk((i, j, k) -> cos(i - j + k))               # signed
    sb = mk((i, j, k) -> sin(i - j))                  # signed
    sa = mk((i, j, k) -> cos(i + j))                  # signed
    g = mk((i, j, k) -> abs(sin(i * j + k)) + 0.05)
    nan_g && (g[2, 3, 1] = NaN)
    _MF.CTTwoAssetSolution{Float64}(bg, ag, V, c, d, sb, sa, g, 3.0, 5.0,
        spzeros(2 * Ib * Ia, 2 * Ib * Ia), true)
end

function _laneF_blanchard()
    ss = _MF.BlanchardOLGSteadyState{Float64}(4.0, 1.2, 0.04, 1.5, 20.0, 0.02, 0.0, true)
    _MF.BlanchardOLGSolution{Float64}(ss, [0.9 0.1; 0.0 1.1],
        ComplexF64[0.9 + 0im, 1.1 + 0im], 0.9, 0.15, true)
end

# One real Reiter HA-DSGE solution, reused across the PLT-34 HADSGESolution tests.
const _LANEF_SPEC = load_ha_example(:krusell_smith)
const _LANEF_SS = compute_steady_state(_LANEF_SPEC; r_bounds=(-0.02, 0.04), max_iter=60, tol=1e-3)
const _LANEF_REITER = solve(_LANEF_SPEC; method=:reiter, ss=_LANEF_SS, n_reduced=12)
const _LANEF_SSJ = solve(_LANEF_SPEC; method=:ssj, ss=_LANEF_SS, T_horizon=30, n_reduced=12)

function _laneF_ks(; agg::Symbol=:K)
    _MF.KrusellSmithSolution{Float64}(_LANEF_SS,
        Dict(agg => [0.087, 0.952, 0.011]), Dict(agg => 0.99987),
        _LANEF_SPEC, true, 5)
end

_laneF_den_haan(; Tn::Int=60, T_burn::Int=10, nan::Bool=false) = begin
    rng = MersenneTwister(7)
    ref = 4.0 .+ 0.1 .* randn(rng, Tn)
    plm = 4.0 .+ 0.1 .* randn(rng, Tn)
    nan && (ref[Tn - 2] = NaN)
    _MF.DenHaanAccuracy{Float64}(:K, 0.42, 0.13, 0.021, 0.020, ref, plm, Tn, T_burn)
end

# =============================================================================

@testset "Lane F — HA-DSGE dynamics + OLG/CT (PLT-34/35)" begin

    # -------------------------------------------------------------------------
    # PLT-35 — CTSteadyState (distribution / policy / lorenz)
    # -------------------------------------------------------------------------
    @testset "PLT-35 CTSteadyState views" begin
        ss = _laneF_ct_ss()

        # :distribution — mass-conserving bar; C7 cap surfaced (25 nodes < 60 bars ⇒
        # no cap here, so use a tiny cap to force the note).
        p = plot_result(ss; view=:distribution)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("Gini", p.html)                       # title carries the Gini
        pc = plot_result(ss; view=:distribution, max_bars=8)
        @test any(t -> occursin("(8 of 25)", t), panel_titles(pc.html))  # Rule 7 cap

        # :policy — two panels, 2 income-state lines each, dash on the 2nd state.
        pp = plot_result(ss; view=:policy)
        check_plot(pp); assert_all_json_valid(pp)
        @test length(panel_titles(pp.html)) == 2             # consumption + saving panels
        @test series_count(pp.html) == 2                     # z1, z2
        @test "z1" in series_names(pp.html) && "z2" in series_names(pp.html)
        @test occursin("6,3", pp.html)                       # dashed 2nd state

        # :lorenz — Lorenz + 45° equality line.
        pl = plot_result(ss; view=:lorenz)
        check_plot(pl); assert_all_json_valid(pl)
        @test "Lorenz Curve" in series_names(pl.html)
        @test "Perfect Equality" in series_names(pl.html)

        # Unknown view → ArgumentError (Rule 3 / C5).
        @test_throws ArgumentError plot_result(ss; view=:bogus)

        # save_path writes a file (Rule 1 / C8).
        tmp = tempname() * ".html"
        p2 = plot_result(ss; save_path=tmp)
        @test p2 isa PlotOutput && isfile(tmp)
        rm(tmp; force=true)
    end

    # -------------------------------------------------------------------------
    # PLT-35 — CTTransition (multi-panel aggregates, var selection, NaN)
    # -------------------------------------------------------------------------
    @testset "PLT-35 CTTransition" begin
        tr = _laneF_ct_transition()
        p = plot_result(tr)
        check_plot(p); assert_all_json_valid(p)
        @test length(panel_titles(p.html)) == 5              # Z, K, r, w, C panels
        @test any(t -> occursin("K transition", t), panel_titles(p.html))
        @test occursin("4,3", p.html)                        # terminal ref line dash

        # var selection by String and Int (Rule 3 / C3).
        pk = plot_result(tr; var="K")
        @test length(panel_titles(pk.html)) == 1
        @test "K" in series_names(pk.html)
        p3 = plot_result(tr; var=3)
        @test "r" in series_names(p3.html)

        @test_throws ArgumentError plot_result(tr; var="nope")   # bad name
        @test_throws ArgumentError plot_result(tr; var=99)       # bad index

        # NaN → null in the drawn series (Rule 4).
        pn = plot_result(_laneF_ct_transition(; nan_K=true))
        assert_nan_becomes_null(pn)
    end

    # -------------------------------------------------------------------------
    # PLT-35 — CTTwoAssetSolution (heatmaps, income select, bad view/index, NaN)
    # -------------------------------------------------------------------------
    @testset "PLT-35 CTTwoAssetSolution" begin
        sol = _laneF_ct_two_asset()
        for vw in (:consumption, :density, :deposit, :liquid_drift, :illiquid_drift)
            p = plot_result(sol; view=vw, income=1)
            check_plot(p); assert_all_json_valid(p)
        end
        # Heatmap color-scale legend + data-driven domain (heatmap rule): the
        # renderer emits a gradient legend and picks scale by view sign.
        pc = plot_result(sol; view=:consumption)
        @test occursin("legend-gradient", pc.html)
        @test occursin("scaleType = \"sequential\"", pc.html)  # consumption ≥ 0
        pd = plot_result(sol; view=:deposit)
        @test occursin("scaleType = \"diverging\"", pd.html)   # deposit is signed
        # row/col labels parse strictly and match the grid dimensions.
        blocks = extract_json_blocks(pc.html)
        rl = assert_strict_json(last(first(filter(b -> b.first == "rowLabels", blocks))))
        cl = assert_strict_json(last(first(filter(b -> b.first == "colLabels", blocks))))
        @test length(rl) == 6 && length(cl) == 7             # Ib × Ia

        @test_throws ArgumentError plot_result(sol; view=:bogus)
        @test_throws ArgumentError plot_result(sol; view=:consumption, income=3)  # >n_inc
        @test_throws ArgumentError plot_result(sol; view=:consumption, income=0)

        # NaN cell → null in the heatmap data (Rule 4).
        pn = plot_result(_laneF_ct_two_asset(; nan_g=true); view=:density)
        assert_nan_becomes_null(pn)
    end

    # -------------------------------------------------------------------------
    # PLT-35 — BlanchardOLGSolution (saddle path)
    # -------------------------------------------------------------------------
    @testset "PLT-35 BlanchardOLGSolution" begin
        p = plot_result(_laneF_blanchard())
        check_plot(p); assert_all_json_valid(p)
        @test any(t -> occursin("Saddle path", t) && occursin("stable eig", t),
                  panel_titles(p.html))
        @test occursin("ellipse", p.html)                    # SS-marking ref circle
        # The SS point + saddle-line overlay live in the scatter `data`/groups blocks.
        blocks = extract_json_blocks(p.html)
        d = assert_strict_json(last(first(filter(b -> b.first == "data", blocks))))
        @test length(d) == 1                                 # single SS marker point
    end

    # -------------------------------------------------------------------------
    # PLT-34 — HADSGESolution (distribution dynamics + inequality, :ssj guard)
    # -------------------------------------------------------------------------
    @testset "PLT-34 HADSGESolution views" begin
        sol = _LANEF_REITER

        # :distribution_dynamics — asset × horizon heatmap of signed deviations,
        # diverging scale, mass-conserving bin-aggregation with a visible cap.
        p = plot_result(sol; horizon=12, max_bins=40)
        check_plot(p); assert_all_json_valid(p)
        @test occursin("scaleType = \"diverging\"", p.html)  # signed deviations
        @test occursin("legend-gradient", p.html)            # heatmap color legend
        @test any(t -> occursin("of 200", t), panel_titles(p.html))  # 40 of 200 nodes (C7)
        blocks = extract_json_blocks(p.html)
        cl = assert_strict_json(last(first(filter(b -> b.first == "colLabels", blocks))))
        @test length(cl) == 12                               # one column per horizon

        # alias :distribution routes to the same view.
        pa = plot_result(sol; view=:distribution, horizon=8)
        @test pa isa PlotOutput

        # :inequality — Gini panel + p10..p90 legended panel, integer horizon ticks.
        pi = plot_result(sol; view=:inequality, horizon=12)
        check_plot(pi); assert_all_json_valid(pi)
        @test length(panel_titles(pi.html)) == 2             # Gini + percentiles panels
        sn = series_names(pi.html)
        @test "Gini" in sn                                   # first panel = Gini
        @test occursin("Wealth Percentile", pi.html)
        @test occursin("d3.format('d')", pi.html)            # integer_x horizon axis

        # Unknown view (Rule 3 / C5).
        @test_throws ArgumentError plot_result(sol; view=:bogus)

        # :ssj has no distribution basis → both views re-raise the informative error.
        @test_throws ErrorException plot_result(_LANEF_SSJ; view=:distribution_dynamics)
        @test_throws ErrorException plot_result(_LANEF_SSJ; view=:inequality)
    end

    # -------------------------------------------------------------------------
    # PLT-34 — KrusellSmithSolution (PLM R² bar + coef annotation, escaping)
    # -------------------------------------------------------------------------
    @testset "PLT-34 KrusellSmithSolution" begin
        p = plot_result(_laneF_ks())
        check_plot(p); assert_all_json_valid(p)
        @test "R²" in series_names(p.html)
        @test occursin("0.952", p.html)                      # PLM coef annotation visible
        @test occursin("orientation", p.html)                # horizontal named-entity bar

        # Escaping round-trip (Rule 5 / A7/A8): a hostile aggregate name reaches the
        # bar category (JSON sink) and the coefficient annotation (HTML title sink).
        ph = plot_result(_laneF_ks(; agg=Symbol(HOSTILE_NAME)))
        assert_escapes(ph)
    end

    # -------------------------------------------------------------------------
    # PLT-34 — DenHaanAccuracy (ref vs PLM lines, error annotation, cap, NaN)
    # -------------------------------------------------------------------------
    @testset "PLT-34 DenHaanAccuracy" begin
        dh = _laneF_den_haan()
        p = plot_result(dh)
        check_plot(p); assert_all_json_valid(p)
        @test "Reference" in series_names(p.html)
        @test "PLM-only" in series_names(p.html)
        @test occursin("6,3", p.html)                        # PLM line dashed (color+dash)
        @test any(t -> occursin("max err", t) && occursin("mean", t),
                  panel_titles(p.html))                      # C9 rounded errors in title

        # Big-N stride cap surfaced (Rule 7): 50 retained points → cap note.
        pc = plot_result(dh; max_points=20)
        @test any(t -> occursin(" of ", t), panel_titles(pc.html))

        # NaN in ref_path → null (Rule 4).
        assert_nan_becomes_null(plot_result(_laneF_den_haan(; nan=true)))
    end
end
