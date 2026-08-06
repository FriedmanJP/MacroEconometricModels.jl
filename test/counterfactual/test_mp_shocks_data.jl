# CF-20 (#400): load_example(:mp_shocks) — US monetary panel with published
# policy-shock series (McKay-Wolf 2023 data appendix).
using Test
using MacroEconometricModels

const MEM = MacroEconometricModels

@testset "mp_shocks dataset (CF-20)" begin
    td = load_example(:mp_shocks)
    @test td isa TimeSeriesData

    names = td.varnames
    @test names == ["ygap", "infl", "ffr", "lpcom", "rr", "mp1", "ad", "bzk_ist"]
    Y = td.data
    @test size(Y) == (240, 8)

    # quarter index helpers (row 1 = 1960Q1)
    row(yr, q) = (yr - 1960) * 4 + q
    colof(n) = findfirst(==(n), names)

    @testset "sample ranges and NaN patterns" begin
        # first/last valid quarter per column (pinned from the source appendix)
        ranges = Dict("ygap" => (row(1969, 1), row(2019, 4)),
                      "infl" => (row(1960, 1), row(2015, 3)),
                      "ffr" => (row(1960, 1), row(2015, 3)),
                      "lpcom" => (row(1960, 1), row(2013, 1)),
                      "rr" => (row(1969, 1), row(2007, 4)),
                      "mp1" => (row(1988, 4), row(2012, 2)),
                      "ad" => (row(1982, 4), row(2008, 3)),
                      "bzk_ist" => (row(1960, 1), row(2012, 1)))
        for (n, (f, l)) in ranges
            v = Y[:, colof(n)]
            @test findfirst(!isnan, v) == f
            @test findlast(!isnan, v) == l
        end
        # NaN outside sample, no all-NaN column
        @test isnan(Y[1, colof("rr")])
        @test isnan(Y[1, colof("mp1")])
        @test count(!isnan, Y[:, colof("rr")]) == 156
    end

    @testset "unit sanity" begin
        ffr = Y[:, colof("ffr")]
        @test ffr[row(1960, 1)] ≈ 3.93 atol = 0.01          # FRED FEDFUNDS 1960Q1
        @test maximum(filter(!isnan, ffr)) > 15.0            # Volcker peak, percent units
        infl = filter(!isnan, Y[:, colof("infl")])
        @test 2.0 < sum(infl) / length(infl) < 5.0           # avg US inflation, % p.a.
    end

    @testset "sanity VAR + conventional sign pattern" begin
        rows = row(1969, 1):row(2007, 4)                     # jointly observed window
        sub = Y[rows, [colof("ygap"), colof("infl"), colof("ffr")]]
        @test !any(isnan, sub)
        m = estimate_var(sub, 4)
        @test m isa VARModel
        ir = irf(m, 16)
        # Cholesky ffr innovation (ordered last): rate up on impact,
        # output gap falls somewhere over the first 4 years (sign-only smoke)
        @test ir.values[1, 3, 3] > 0
        @test minimum(ir.values[:, 1, 3]) < 0
    end

    @testset "refs and metadata" begin
        s = sprint(refs, td)
        @test occursin("Romer", s)
        @test occursin("Gertler", s)
        @test occursin("McKay", s)
        @test occursin("Ben Zeev", s)
        @test occursin("Wieland", s)
        @test occursin("Aruoba", s)
        @test !isempty(td.desc)
        @test occursin("NaN", td.vardesc["rr"])
    end
end
