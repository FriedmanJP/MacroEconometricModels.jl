using Test, MacroEconometricModels

@testset "Parser extensions" begin
    if Base.find_package("ZipFile") !== nothing
        @eval using ZipFile
        ext = Base.get_extension(MacroEconometricModels, :MacroEconometricModelsZipFileExt)
        @test ext !== nothing
        # build a tiny zip with one csv and parse it
        dir = mktempdir(); zp = joinpath(dir, "t.zip")
        w = ZipFile.Writer(zp); f = ZipFile.addfile(w, "io.csv")
        print(f, "150.0,500.0,350.0\n200.0,100.0,1700.0\n"); close(w)
        io = MacroEconometricModels._parse_zip_io(zp; n_sectors=2, n_fd=1, member="io.csv")
        @test io.x ≈ [1000.0, 2000.0]
        # G-15 (#254): zip-bomb guard — a member declaring more than max_uncompressed
        # bytes is refused before it is read into memory.
        @test_throws ErrorException MacroEconometricModels._parse_zip_io(
            zp; n_sectors=2, n_fd=1, member="io.csv", max_uncompressed=3)
    else
        @test_broken false  # ZipFile not installed in this environment
    end

    if Base.find_package("XLSX") !== nothing
        @eval using XLSX
        ext = Base.get_extension(MacroEconometricModels, :MacroEconometricModelsXLSXExt)
        @test ext !== nothing
        dir = mktempdir(); xp = joinpath(dir, "t.xlsx")
        XLSX.openxlsx(xp, mode="w") do xf
            s = xf[1]
            s["A1"] = 150.0; s["B1"] = 500.0; s["C1"] = 350.0
            s["A2"] = 200.0; s["B2"] = 100.0; s["C2"] = 1700.0
        end
        io = MacroEconometricModels._parse_xlsx_io(xp; n_sectors=2, n_fd=1)
        @test io.x ≈ [1000.0, 2000.0]
    else
        @test_broken false  # XLSX not installed in this environment
    end
end
