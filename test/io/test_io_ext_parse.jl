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
    else
        @test_broken false  # ZipFile not installed in this environment
    end
end
