using Test, MacroEconometricModels, LinearAlgebra
using MacroEconometricModels: nsectors, nregions, _parse_icio_csv_text, _parse_wiod_matrix

const FIX = joinpath(@__DIR__, "fixtures")

# ── OECD ICIO ────────────────────────────────────────────────────────────────

@testset "parse_icio CSV labeled MRIO" begin
    path = joinpath(FIX, "icio_toy_2010.csv")
    io = parse_icio(path; year=2010, check=true)

    @test io.source == "OECD ICIO"
    @test io.year == 2010
    @test nregions(io) == 2
    @test nsectors(io) == 2
    @test io.regions == ["USA", "CHN"]
    @test io.sectors == ["USA_S1", "USA_S2", "CHN_S1", "CHN_S2"]
    @test io.fd_cats == ["USA_HFCE", "CHN_HFCE"]
    @test io.va_cats == ["VA"]
    @test size(io.Z) == (4, 4)
    @test size(io.Y) == (4, 2)
    @test io.Z[1, 1] ≈ 10.0
    @test io.Y[1, 1] ≈ 20.0
    @test io.x ≈ [38.0, 45.0, 48.0, 36.0] atol=1e-12
    @test io.va ≈ reshape([24.0, 24.0, 32.0, 21.0], 1, 4) atol=1e-12
    @test io.unit == "Million USD"

    # MRIO block accessors work on the labeled table.
    @test region_block(io, "USA", "USA") ≈ [10.0 5.0; 3.0 15.0] atol=1e-12
    @test region_block(io, "USA", "CHN") ≈ [1.0 0.0; 0.0 1.0] atol=1e-12
    bt = bilateral_trade(io, "USA", "CHN")
    @test bt.intermediate ≈ 2.0 atol=1e-12   # 1+0+0+1
    @test bt.final ≈ 3.0 atol=1e-12          # 2+1
end

@testset "parse_icio CN/MX aggregation" begin
    path = joinpath(FIX, "icio_cnmx_2010.csv")
    io = parse_icio(path; year=2010, aggregate_cn_mx=true, check=false)
    @test io.regions == ["USA", "CHN"]
    @test io.sectors == ["USA_S1", "CHN_S1"]
    # CN1_S1 + CN2_S1 intermediate into USA: 1+0 = 1
    @test io.Z[2, 1] ≈ 1.0 atol=1e-12
    # Domestic CHN block: CN1→CN1 8 + CN1→CN2 2 + CN2→CN1 1 + CN2→CN2 6 = 17
    @test io.Z[2, 2] ≈ 17.0 atol=1e-12
    # Destination FD columns are merged so Y stays region-blocked (G=2, 1 cat).
    @test size(io.Y, 2) == 2
    @test io.fd_cats == ["USA_HFCE", "CHN_HFCE"]
    @test io.Y ≈ [20.0 4.0; 3.0 30.0] atol=1e-12   # CHN_HFCE = CN1+CN2

    # Final-export routes remain visible to MRIO accounting.
    bt = bilateral_trade(io, "USA", "CHN")
    @test bt.intermediate ≈ 3.0 atol=1e-12   # USA→CN1 2 + USA→CN2 1
    @test bt.final ≈ 4.0 atol=1e-12          # 3+1
    @test gross_exports(io, "USA") ≈ [7.0] atol=1e-12
    ed = export_decomposition(io, "USA")
    @test ed.gross_exports ≈ 7.0 atol=1e-10
    @test ed.dva + ed.rdv + ed.fva + ed.pdc ≈ ed.gross_exports atol=1e-10
    # Term 1 is DVA in final-goods exports — zero if FD were left unblocked.
    @test ed.terms[1] > 0

    io_raw = parse_icio(path; year=2010, aggregate_cn_mx=false, check=false)
    @test nregions(io_raw) == 3
    @test "CN1" in io_raw.regions
    @test "CN2" in io_raw.regions
end

@testset "parse_icio from text + year-from-name" begin
    text = read(joinpath(FIX, "icio_toy_2010.csv"), String)
    io = _parse_icio_csv_text(text; year=nothing, check=true)
    @test nregions(io) == 2
    @test io.x ≈ [38.0, 45.0, 48.0, 36.0]

    # Year inference: ICIO2023_2016.csv → 2016
    @test MacroEconometricModels._icio_year_from_name("ICIO2023_2016.csv") == 2016
    @test MacroEconometricModels._icio_year_from_name("ICIO2016_2005.csv") == 2005
    @test MacroEconometricModels._icio_year_from_name("no_year.csv") === nothing
end

@testset "parse_icio directory selection + errors" begin
    dir = mktempdir()
    cp(joinpath(FIX, "icio_toy_2010.csv"), joinpath(dir, "ICIO2016_2010.csv"))
    io = parse_icio(dir; year=2010, check=false)
    @test io.year == 2010
    @test nregions(io) == 2

    @test_throws ArgumentError parse_icio(dir)  # year required
    @test_throws ArgumentError parse_icio(joinpath(dir, "missing.csv"))
    @test_throws ArgumentError parse_icio(joinpath(dir, "x.foo"))
end

@testset "parse_icio zip (ZipFile extension)" begin
    if Base.find_package("ZipFile") === nothing
        @test_broken false
    else
        @eval using ZipFile
        ext = Base.get_extension(MacroEconometricModels, :MacroEconometricModelsZipFileExt)
        @test ext !== nothing

        dir = mktempdir()
        zp = joinpath(dir, "ICIO_v2016_2010.zip")
        csv = read(joinpath(FIX, "icio_toy_2010.csv"), String)
        w = ZipFile.Writer(zp)
        f = ZipFile.addfile(w, "ICIO2016_2010.csv")
        print(f, csv)
        close(w)

        io = parse_icio(zp; year=2010, check=true)
        @test nregions(io) == 2
        @test io.x ≈ [38.0, 45.0, 48.0, 36.0]

        # Explicit member=
        io2 = parse_icio(zp; member="ICIO2016_2010.csv", check=false)
        @test io2.Z ≈ io.Z

        @test_throws ArgumentError parse_icio(zp; member="nope.csv")
    end
end

# ── WIOD WIOT ────────────────────────────────────────────────────────────────

"Build a 2-region × 2-sector WIOT-like Any matrix (pymrio header layout)."
function _wiod_toy_matrix()
    # Full sheet (1-based), 11 rows × 11 cols:
    #   cols: 1=code, 2=name, 3=region, 4=c_code,
    #         5-8 = USA_c1,USA_c2,CHN_c1,CHN_c2,
    #         9-10 = USA_c37, CHN_c37 (FD), 11 = TOTAL
    # Rows 1-2 are the empty top rows pymrio drops; meta lives in col 1 of
    # rows 1 (year), 3 (iosystem), 4 (unit). Header block is rows 3-6.
    nr, nc = 11, 11
    A = Array{Any}(missing, nr, nc)

    # Column headers at rows 3-6 (become rows 1-4 after dropping 1-2).
    # code row
    A[3, 2] = "code"; A[3, 3] = "code"; A[3, 4] = "code"
    A[3, 5] = "AtB"; A[3, 6] = "MtB"; A[3, 7] = "AtB"; A[3, 8] = "MtB"
    A[3, 9] = "CONS"; A[3, 10] = "CONS"; A[3, 11] = "TOT"
    # sector names
    A[4, 2] = "sector"; A[4, 3] = "sector"; A[4, 4] = "sector"
    A[4, 5] = "Agriculture"; A[4, 6] = "Manufacturing"
    A[4, 7] = "Agriculture"; A[4, 8] = "Manufacturing"
    A[4, 9] = "Consumption"; A[4, 10] = "Consumption"; A[4, 11] = "Total"
    # region
    A[5, 2] = "region"; A[5, 3] = "region"; A[5, 4] = "region"
    A[5, 5] = "USA"; A[5, 6] = "USA"; A[5, 7] = "CHN"; A[5, 8] = "CHN"
    A[5, 9] = "USA"; A[5, 10] = "CHN"; A[5, 11] = "TOT"
    # c_code
    A[6, 2] = "c_code"; A[6, 3] = "c_code"; A[6, 4] = "c_code"
    A[6, 5] = "c1"; A[6, 6] = "c2"; A[6, 7] = "c1"; A[6, 8] = "c2"
    A[6, 9] = "c37"; A[6, 10] = "c37"; A[6, 11] = "TOTAL"

    # Meta in col 1 — written last so it is not overwritten by headers.
    A[1, 1] = "WIOT for 2010"
    A[3, 1] = "(industry-by-industry)"
    A[4, 1] = "(millions of US\$)"

    # Industry rows 7-10. Row labels in cols 1-4. Values match icio_toy.
    labs = [
        ("AtB", "Agriculture", "USA", "c1", [10, 5, 1, 0, 20, 2, 38]),
        ("MtB", "Manufacturing", "USA", "c2", [3, 15, 0, 1, 25, 1, 45]),
        ("AtB", "Agriculture", "CHN", "c1", [0, 1, 12, 4, 1, 30, 48]),
        ("MtB", "Manufacturing", "CHN", "c2", [1, 0, 3, 10, 2, 20, 36]),
    ]
    for (r, (code, name, reg, cc, vals)) in enumerate(labs)
        i = 6 + r
        A[i, 1] = code; A[i, 2] = name; A[i, 3] = reg; A[i, 4] = cc
        for (k, v) in enumerate(vals)
            A[i, 4 + k] = Float64(v)
        end
    end
    # VA row
    A[11, 1] = "VA"; A[11, 2] = "Value added"; A[11, 3] = ""; A[11, 4] = "r45"
    for (k, v) in enumerate([24, 24, 32, 21, 0, 0, 101])
        A[11, 4 + k] = Float64(v)
    end
    return A
end

@testset "parse_wiod matrix core (no XLSX required)" begin
    A = _wiod_toy_matrix()
    # After dropping top 2 rows, headers are at 1..4 and last c2 is at col/row 8
    # in the *original* matrix (cols 5-8 industries). last_z after drop:
    # original row 10 has c2 for CHN_MtB → after drop rows 1-2, that is row 8.
    # last_interind_code = c2
    io = _parse_wiod_matrix(A; year=2010, last_interind_code="c2", check=true)

    @test io.source == "WIOD 2013"
    @test io.year == 2010
    @test nregions(io) == 2
    @test nsectors(io) == 2
    @test io.regions == ["USA", "CHN"]
    @test io.sectors == ["USA_AtB", "USA_MtB", "CHN_AtB", "CHN_MtB"]
    @test size(io.Z) == (4, 4)
    @test io.Z[1, 1] ≈ 10.0
    @test io.Y[1, 1] ≈ 20.0
    @test io.x ≈ [38.0, 45.0, 48.0, 36.0] atol=1e-12
    @test io.va_cats == ["Value added"]
    @test occursin("million", lowercase(io.unit)) || occursin("US", io.unit)

    @test region_block(io, "USA", "CHN") ≈ [1.0 0.0; 0.0 1.0] atol=1e-12
end

@testset "parse_wiod xlsx (XLSX extension)" begin
    if Base.find_package("XLSX") === nothing
        @test_broken false
    else
        @eval using XLSX
        ext = Base.get_extension(MacroEconometricModels, :MacroEconometricModelsXLSXExt)
        @test ext !== nothing

        dir = mktempdir()
        xp = joinpath(dir, "wiot10_row_toy.xlsx")
        A = _wiod_toy_matrix()
        XLSX.openxlsx(xp, mode="w") do xf
            s = xf[1]
            XLSX.rename!(s, "WIOT")
            for i in 1:size(A, 1), j in 1:size(A, 2)
                v = A[i, j]
                (v === missing || v === nothing) && continue
                s[i, j] = v
            end
        end

        io = parse_wiod(xp; year=2010, last_interind_code="c2", check=true)
        @test nregions(io) == 2
        @test io.x ≈ [38.0, 45.0, 48.0, 36.0] atol=1e-12

        # Directory + year selection
        io2 = parse_wiod(dir; year=2010, last_interind_code="c2", check=false)
        @test io2.Z ≈ io.Z

        @test MacroEconometricModels._wiod_year_from_name("wiot09_row_apr12.xlsx") == 2009
        @test MacroEconometricModels._wiod_year_from_name("wiot10.xlsx") == 2010

        @test_throws ArgumentError parse_wiod(dir)  # year required
        @test_throws ArgumentError parse_wiod(joinpath(dir, "nope.xlsx"))
    end
end

@testset "parse_wiod / parse_icio error stubs without optional packages" begin
    # Zip path without ZipFile loaded → actionable error from _zip_member_names
    # (only when the extension is not already active from a prior testset).
    if Base.get_extension(MacroEconometricModels, :MacroEconometricModelsZipFileExt) === nothing
        dir = mktempdir(); zp = joinpath(dir, "t.zip"); touch(zp)
        err = try
            parse_icio(zp)
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("ZipFile", sprint(showerror, err))
    end
end
