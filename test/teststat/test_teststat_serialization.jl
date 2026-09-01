# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

if !@isdefined(_assert_roundtrip)
    include(joinpath(@__DIR__, "..", "serialization_helpers.jl"))
end

@testset "RSER-02 JohansenResult serialization" begin
    Yci = cumsum(randn(MersenneTwister(3), 100, 2); dims=1)
    jr = johansen_test(Yci, 2)
    jr2 = _assert_roundtrip(jr)
    _assert_report_equal(jr, jr2)
    @test plot_result(jr2) isa PlotOutput
    @test sprint(io -> refs(io, jr)) == sprint(io -> refs(io, jr2))
end

# 47 RSER-09 types. GrangerCausalityResult / VARStationarityResult / JohansenResult
# were registered in RSER-02/03 (Task 16) and are not listed here.
const _RSER09 = ("ADF2BreakResult", "ADFResult", "AndrewsResult", "BDSResult",
                 "BaiPerronResult", "BoxPierceResult", "BreitungPanelResult",
                 "BubbleResult", "CorTestResult", "DFGLSResult",
                 "DumitrescuHurlinResult", "DurbinWatsonResult", "EDFTestResult",
                 "ERSResult", "EngleGrangerResult", "EqualityTestResult",
                 "FactorBreakResult", "FisherJohansenResult", "FisherPanelResult",
                 "FourierADFResult", "FourierKPSSResult", "GregoryHansenResult",
                 "HEGYResult", "HadriResult", "HansenInstabilityResult",
                 "IPSResult", "KPSSResult", "KaoResult", "LLCResult",
                 "LMTestResult", "LMUnitRootResult", "LRTestResult",
                 "LjungBoxResult", "MoonPerronResult", "NgPerronResult",
                 "NormalityTestResult", "NormalityTestSuite", "PANICResult",
                 "PPResult", "PanelUnitRootSummary", "ParkAddedResult",
                 "PedroniResult", "PesaranCIPSResult", "PhillipsOuliarisResult",
                 "VarianceRatioResult", "WesterlundResult", "ZAResult")

_rser09_cv(a, b, c) = Dict(1 => Float64(a), 5 => Float64(b), 10 => Float64(c))
_rser09_np_cv() = Dict(:MZa => _rser09_cv(-13.8, -8.1, -5.7),
                       :MZt => _rser09_cv(-2.58, -1.98, -1.62),
                       :MSB => _rser09_cv(0.17, 0.23, 0.28),
                       :MPT => _rser09_cv(1.78, 3.17, 4.45))

function _rser09_fixtures()
    cv = _rser09_cv(-3.43, -2.86, -2.57)
    npcv = _rser09_np_cv()
    llc = LLCResult(-1.8, 0.04, -2.1, -0.05, 1.1, -0.55, 0.85, 38.0,
                    [1, 1, 0, 1, 0, 1], :constant, 40, 6)
    ips = IPSResult(-1.7, 0.04, -1.9, fill(-1.8, 6), -1.53, 0.74,
                    [1, 1, 0, 1, 0, 1], :constant, 40, 6)
    breit = BreitungPanelResult(-1.5, 0.07, 0, :constant, 40, 6)
    fish = FisherPanelResult(18.0, 0.11, 18.0, 0.11, -0.4, 0.34,
                             -0.5, 0.31, 0.3, 0.38, fill(0.2, 6),
                             :adf, :mw, 40, 6)
    had = HadriResult(1.2, 0.11, 0.2, 1 / 6, sqrt(1 / 45), true, :constant, 40, 6)
    panic = PANICResult([-3.5], [0.01], 2.1, 0.03, fill(-2.0, 6), fill(0.1, 6),
                        1, :pooled, 40, 6)
    cips = PesaranCIPSResult(-2.5, 0.04, fill(-1.8, 6),
                             Dict(1 => -2.88, 5 => -2.32, 10 => -2.07),
                             1, :constant, 40, 6)
    mp = MoonPerronResult(-2.1, -1.9, 0.02, 0.03, 1, 40, 6)
    jb = NormalityTestResult(:jarque_bera, 3.2, 0.20, 4, 2, 40, nothing, nothing)
    jb_comp = NormalityTestResult(:jarque_bera, 2.1, 0.35, 4, 2, 40,
                                  [1.0, 1.1], [0.4, 0.3])

    return Pair{String,Any}[
        "ADF2BreakResult" => ADF2BreakResult(-4.2, 0.08, 20, 35, 0.4, 0.7, 0, :level, cv, 50),
        "ADFResult" => ADFResult(-2.5, 0.12, 1, :constant, cv, 58),
        "AndrewsResult" => AndrewsResult(3.45, 0.02, 50, 0.5, :supwald,
                                        Dict(1 => 12.0, 5 => 8.5, 10 => 6.8),
                                        fill(2.0, 20), 0.15, 60, 2),
        "BDSResult" => BDSResult([2], [0.5], [0.7], 1.0, reshape([1.2], 1, 1),
                                reshape([0.23], 1, 1), reshape([NaN], 1, 1),
                                reshape([0.4], 1, 1), 60, true, 0, 1234),
        "BaiPerronResult" => BaiPerronResult(2, [30, 70], [(25, 35), (65, 75)],
                                            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                                            [10.0, 8.0], [0.01, 0.03],
                                            [9.0], [0.02],
                                            [100.0, 90.0, 85.0], [100.0, 92.0, 88.0],
                                            0.15, 100),
        "BoxPierceResult" => BoxPierceResult(8.1, 0.09, 4, 4, 60),
        "BreitungPanelResult" => breit,
        "BubbleResult" => BubbleResult(:sadf, 1.2, 0.3, Dict(1 => 2.0, 5 => 1.5, 10 => 1.2),
                                       [0.1, 0.8, 1.2], [1.0, 1.0, 1.0], [10, 11, 12],
                                       [(10, 12)], 0.4, 0, :asymptotic, 10, 20),
        "CorTestResult" => CorTestResult(:pearson, 0.4, 2.1, 0.04, 40, 38.0,
                                         0.1, 0.6, false, "Fisher z"),
        "DFGLSResult" => DFGLSResult(-2.1, 0.15, 3.2, 0.20, -8.0, -1.9, 0.24, 3.5,
                                     1, :constant, cv, _rser09_cv(1.9, 3.2, 4.0), npcv, 58),
        "DumitrescuHurlinResult" => DumitrescuHurlinResult(3.2, 1.1, 0.14, 0.9, 0.18,
                                                          fill(3.0, 6), 1, 6, 29, 0, 0, 1234,
                                                          NaN, :x, :y),
        "DurbinWatsonResult" => DurbinWatsonResult(1.95, 0.40, 60),
        "EDFTestResult" => EDFTestResult(:ad, :normal, :estimate, 0.4, 0.35, 0.25, 60,
                                         [0.0, 1.0], cv, "estimated normal"),
        "ERSResult" => ERSResult(3.1, 0.18, :constant, _rser09_cv(1.9, 3.2, 4.0), 60),
        "EngleGrangerResult" => EngleGrangerResult(-3.1, 0.09, 1, :constant, 1, 2, 60),
        "EqualityTestResult" => EqualityTestResult(:two_sample_t, 1.4, 0.16, 38.0, NaN,
                                                   2, [20, 20], false, "Welch"),
        "FactorBreakResult" => FactorBreakResult(3.2, 0.05, nothing, :chen_dolado_gonzalo,
                                                 1, 40, 6),
        "FisherJohansenResult" => FisherJohansenResult([0, 1], [12.0, 3.0], [0.02, 0.40],
                                                       [10.0, 2.0], [0.03, 0.50],
                                                       [0.1 0.4; 0.2 0.5; 0.15 0.3; 0.12 0.45],
                                                       [0.1 0.4; 0.2 0.5; 0.15 0.3; 0.12 0.45],
                                                       :mw, :constant, 1, 1, 4, 2),
        "FisherPanelResult" => fish,
        "FourierADFResult" => FourierADFResult(-3.0, 0.10, 1, 5.0, 0.08, 0, :constant,
                                               cv, _rser09_cv(6.0, 4.0, 3.0), 50),
        "FourierKPSSResult" => FourierKPSSResult(0.08, 0.12, 1, 4.5, 0.09, :constant,
                                                 Dict(1 => 0.22, 5 => 0.15, 10 => 0.12),
                                                 _rser09_cv(6.0, 4.0, 3.0), 2, 50),
        "GregoryHansenResult" => GregoryHansenResult(-4.5, 0.08, -4.4, 0.09, -28.0, 0.10,
                                                     20, 22, 21, :C, 1, cv,
                                                     Dict(1 => -40.0, 5 => -32.0, 10 => -28.0),
                                                     50),
        "HEGYResult" => HEGYResult(4, :const_trend_seas, 0, [-2.0, -1.5, 0.3, 0.2],
                                   -2.0, -1.5, cv, cv, [π / 2], [4.0],
                                   Dict(1 => 8.0, 5 => 6.0, 10 => 5.0), 5.0, 6.0, 70),
        "HadriResult" => had,
        "HansenInstabilityResult" => HansenInstabilityResult(0.4, 0.12, :constant, :const, 2, 1, 60),
        "IPSResult" => ips,
        "KPSSResult" => KPSSResult(0.15, 0.10, :constant, Dict(1 => 0.74, 5 => 0.46, 10 => 0.35),
                                   3, 60),
        "KaoResult" => KaoResult(["DFrho", "DFt", "DFrho_star", "DFt_star", "ADF"],
                                 [-1.2, -1.3, -1.1, -1.0, -1.4],
                                 [0.11, 0.10, 0.13, 0.16, 0.08],
                                 0.85, -1.3, -1.4, 0.5, 0.6, 1, 2, 1, 30, 6),
        "LLCResult" => llc,
        "LMTestResult" => LMTestResult(4.2, 0.04, 1, 80, 0.3),
        "LMUnitRootResult" => LMUnitRootResult(-2.8, 0.12, 0, Int[], Float64[], 0, :level, cv, 50),
        "LRTestResult" => LRTestResult(5.1, 0.02, 1, -120.0, -117.45, 2, 3, 80, 80),
        "LjungBoxResult" => LjungBoxResult(9.2, 0.06, 4, 4, 60),
        "MoonPerronResult" => mp,
        "NgPerronResult" => NgPerronResult(-8.0, -1.9, 0.24, 3.5, :constant, npcv, 58),
        "NormalityTestResult" => jb,
        "NormalityTestSuite" => NormalityTestSuite([jb, jb_comp], randn(40, 2), 2, 40),
        "PANICResult" => panic,
        "PPResult" => PPResult(-2.4, 0.14, :constant, cv, 3, 59),
        "PanelUnitRootSummary" => PanelUnitRootSummary(panic, cips, mp, llc, ips, breit,
                                                       fish, had, String[]),
        "ParkAddedResult" => ParkAddedResult(3.5, 0.06, 1, 0, :constant, :const, 1, 60),
        "PedroniResult" => PedroniResult(["panel-v", "panel-rho", "panel-t", "panel-adf",
                                          "group-rho", "group-t", "group-adf"],
                                         [1.2, -1.1, -1.3, -1.5, -1.0, -1.2, -1.4],
                                         [1.1, -1.0, -1.2, -1.4, -0.9, -1.1, -1.3],
                                         [0.14, 0.16, 0.11, 0.08, 0.18, 0.13, 0.10],
                                         fill(0.5, 7), fill(1.0, 7), :constant, 1, 2, 2, 30, 6),
        "PesaranCIPSResult" => cips,
        "PhillipsOuliarisResult" => PhillipsOuliarisResult(-2.8, 0.11, -12.0, 0.15,
                                                           :constant, :bartlett, 3.0, 1, 2, 60),
        "VarianceRatioResult" => VarianceRatioResult([2, 4], [0.9, 0.8], [0.5, 0.8],
                                                     [0.4, 0.7], [0.62, 0.42], [0.69, 0.48],
                                                     0.8, 0.42, 0.7, 0.48, :lomackinlay, true, false,
                                                     Float64[], Float64[], Float64[],
                                                     Float64[], Float64[], Float64[],
                                                     0, :rademacher, 1234, Float64[], NaN, 60),
        "WesterlundResult" => WesterlundResult(["Gt", "Ga", "Pt", "Pa"],
                                               [-2.0, -4.0, -3.0, -5.0],
                                               [-1.2, -1.0, -1.3, -1.1],
                                               [0.11, 0.16, 0.10, 0.14],
                                               fill(NaN, 4), :constant, 1, 1, 0, 2, 0, 1234, 30, 6),
        "ZAResult" => ZAResult(-4.1, 0.08, 25, 0.5, :constant, cv, 0, 50),
    ]
end

@testset "RSER-09 test-statistic serialization (#782)" begin
    @testset "registry" begin
        @test length(_RSER09) == 47
        for name in _RSER09
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
        @test !any(v == "pending RSER-09" for v in values(_MEM._SERIALIZATION_EXCLUDED))
        for name in ("GrangerCausalityResult", "VARStationarityResult", "JohansenResult")
            @test haskey(_MEM._SERIALIZABLE_TYPES, name)
            @test !haskey(_MEM._SERIALIZATION_EXCLUDED, name)
        end
    end

    fixtures = _rser09_fixtures()
    @test length(fixtures) == 47
    @test Set(first.(fixtures)) == Set(_RSER09)

    byname = Dict{String,Any}(fixtures)

    @testset "generic reconstruct + roundtrip + report ($name)" for (name, r) in fixtures
        Tw = Base.typename(typeof(r)).wrapper
        @test string(nameof(Tw)) == name
        @test _from_serializable_is_generic(Tw)
        r2 = _assert_roundtrip(r)
        _assert_consumers(r, r2)
    end

    @testset "NgPerron / DFGLS nested Dict of arrays" begin
        np = byname["NgPerronResult"]
        np2 = _roundtrip(np)
        @test np2.critical_values isa AbstractDict
        @test Set(keys(np2.critical_values)) == Set(keys(np.critical_values))
        @test all(v isa AbstractDict for v in values(np2.critical_values))
        df = byname["DFGLSResult"]
        df2 = _roundtrip(df)
        @test df2.mgls_critical_values isa AbstractDict
        @test Set(keys(df2.mgls_critical_values)) == Set(keys(df.mgls_critical_values))
        @test all(v isa AbstractDict for v in values(df2.mgls_critical_values))
    end

    @testset "BaiPerron Vector{Tuple} / nested vectors" begin
        bp = byname["BaiPerronResult"]
        @test bp.break_cis isa Vector{<:Tuple}
        @test bp.regime_coefs isa Vector{<:AbstractVector}
        bp2 = _roundtrip(bp)
        @test bp2.break_cis isa Vector{<:Tuple}
        @test bp2.break_cis == bp.break_cis
        @test bp2.regime_coefs isa Vector{<:AbstractVector}
        @test bp2.regime_coefs == bp.regime_coefs
    end

    @testset "BubbleResult episodes Vector{Tuple}" begin
        br = byname["BubbleResult"]
        @test br.episodes isa Vector{<:Tuple}
        br2 = _roundtrip(br)
        @test br2.episodes isa Vector{<:Tuple}
        @test br2.episodes == br.episodes
        constructed = BubbleResult(:sadf, 1.2, 0.3, Dict(1 => 2.0, 5 => 1.5, 10 => 1.2),
                                   [0.1, 0.8, 1.2], [1.0, 1.0, 1.0], [10, 11, 12],
                                   [(10, 12)], 0.4, 0, :asymptotic, 10, 20)
        c2 = _assert_roundtrip(constructed)
        @test c2.episodes == [(10, 12)]
        _assert_consumers(constructed, c2)
    end

    @testset "FactorBreakResult Union{Nothing,Vector}" begin
        fb = byname["FactorBreakResult"]
        fb2 = _roundtrip(fb)
        @test fb2.series_statistics === fb.series_statistics
        @test fb2.series_break_dates === fb.series_break_dates
        with_series = FactorBreakResult(1.1, 0.04, 12, :breitung_eickmeier, 1, 40, 6,
                                        [1.0, 2.0, 0.5], [10, 12, 15])
        ws2 = _assert_roundtrip(with_series)
        @test ws2.series_statistics isa Vector
        @test ws2.series_break_dates isa Vector{Int}
        @test ws2.break_date == 12
        _assert_report_equal(with_series, ws2)
    end

    @testset "PanelUnitRootSummary all eight sub-results" begin
        s = byname["PanelUnitRootSummary"]
        @test s.panic isa PANICResult
        @test s.cips isa PesaranCIPSResult
        @test s.moon_perron isa MoonPerronResult
        @test s.llc isa LLCResult
        @test s.ips isa IPSResult
        @test s.breitung isa BreitungPanelResult
        @test s.fisher isa FisherPanelResult
        @test s.hadri isa HadriResult
        s2 = _roundtrip(s)
        @test s2 isa PanelUnitRootSummary
        @test s2.panic isa PANICResult
        @test s2.cips isa PesaranCIPSResult
        @test s2.moon_perron isa MoonPerronResult
        @test s2.llc isa LLCResult
        @test s2.ips isa IPSResult
        @test s2.breitung isa BreitungPanelResult
        @test s2.fisher isa FisherPanelResult
        @test s2.hadri isa HadriResult
        @test s2.errors == s.errors
        payload = _MEM._to_serializable(s)
        @test payload["panic"]["__struct__"] == "PANICResult"
        @test payload["hadri"]["__struct__"] == "HadriResult"
    end

    @testset "NormalityTestSuite nested results" begin
        suite = byname["NormalityTestSuite"]
        @test suite.results isa Vector{<:NormalityTestResult}
        s2 = _roundtrip(suite)
        @test s2.results isa Vector{<:NormalityTestResult}
        @test length(s2.results) == length(suite.results)
        @test s2.results[1].test_name == suite.results[1].test_name
    end

    @testset "disk JLD2 BaiPerron / PanelUnitRootSummary / Bubble" begin
        dir = mktempdir()
        for (key, fname) in (("BaiPerronResult", "baiperron.jld2"),
                             ("PanelUnitRootSummary", "panel_ur.jld2"),
                             ("BubbleResult", "bubble.jld2"))
            r = byname[key]
            _MEM._assert_plain_payload(_MEM._build_container(r))
            path = joinpath(dir, fname)
            save_model(r, path)
            r3 = load_model(path)
            @test typeof(r3).name === typeof(r).name
            _assert_report_equal(r, r3)
            applicable(plot_result, r) && _assert_plot_equal(r, r3)
        end
    end
end
