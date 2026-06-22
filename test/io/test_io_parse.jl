using Test, MacroEconometricModels, DelimitedFiles
using MacroEconometricModels: _parse_csv_io

@testset "parse_io CSV core + extension errors" begin
    dir = mktempdir()
    # square Z (2×2) with a final-demand column, written as a csv block
    csv = joinpath(dir, "io.csv")
    writedlm(csv, [150.0 500.0 350.0; 200.0 100.0 1700.0], ',')
    io = _parse_csv_io(csv; n_sectors=2, n_fd=1)
    @test io.x ≈ [1000.0, 2000.0]

    # parse_io dispatches .csv to the in-core parser
    io_disp = parse_io(csv; source=:oecd, n_sectors=2, n_fd=1)
    @test io_disp.x ≈ [1000.0, 2000.0]

    # parsing a .zip without ZipFile loaded errors helpfully
    z = joinpath(dir, "x.zip"); touch(z)
    err = try
        parse_io(z; source=:oecd)
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("ZipFile", sprint(showerror, err))

    # parsing a .xlsx without XLSX loaded errors helpfully (runs before the
    # ext_parse test loads XLSX in the full suite)
    xf = joinpath(dir, "x.xlsx"); touch(xf)
    err2 = try
        parse_io(xf; source=:wiod)
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("XLSX", sprint(showerror, err2))

    # tsv via delim + the n_fd default
    tsv = joinpath(dir, "io.tsv")
    writedlm(tsv, [150.0 500.0 350.0; 200.0 100.0 1700.0], '\t')
    @test _parse_csv_io(tsv; n_sectors=2, delim='\t').x ≈ [1000.0, 2000.0]

    # unsupported extension
    @test_throws ArgumentError parse_io(joinpath(dir, "x.foo"); source=:oecd)
end
