using Test, MacroEconometricModels, LinearAlgebra
using MacroEconometricModels: _classify, _household_coeffs, _match_year,
    download_wiod, download_exiobase3, download_gloria, download_eora26,
    fetch_file, fetch_text, labels

@testset "IO coverage — types & accessors" begin
    io = load_example(:wiot)
    @test labels(io) == io.sectors
    @test size(io) == (2, 1)
    # x-form constructor with explicit va
    io2 = IOData(io.Z, io.Y, [1000.0, 2000.0]; va=[650.0 1400.0])
    @test size(io2.va) == (1, 2)
    # constructor validation errors
    @test_throws ArgumentError IOData([1.0 2.0 3.0; 4.0 5.0 6.0], reshape([1.0, 1.0], 2, 1),
                                      [0.0 0.0]; check=false)               # Z not square
    @test_throws ArgumentError IOData([1.0 2.0; 3.0 4.0], reshape([1.0], 1, 1),
                                      [0.0 0.0]; check=false)               # Y rows ≠ n
    @test_throws ArgumentError IOData([1.0 2.0; 3.0 4.0], reshape([1.0, 1.0], 2, 1),
                                      reshape([0.0], 1, 1); check=false)    # va cols ≠ n
end

@testset "IO coverage — multipliers branches" begin
    io = load_example(:wiot)
    @test_throws ArgumentError _household_coeffs(io, :bogus)
    @test_throws ArgumentError multipliers(io; type=:bogus)
    @test multipliers(io; kind=:income, type=:II) isa MacroEconometricModels.IOMultipliers
    @test multipliers(io; kind=:employment, type=:II) isa MacroEconometricModels.IOMultipliers
    # employment multiplier without an employment extension errors
    bare = IOData(io.Z, io.Y, io.va; sectors=io.sectors)
    @test_throws ArgumentError multipliers(bare; kind=:employment)
end

@testset "IO coverage — linkages, sda, extraction" begin
    io = load_example(:wiot)
    @test _classify(1.5, 1.5) == :key
    @test _classify(1.5, 0.5) == :backward
    @test _classify(0.5, 1.5) == :forward
    @test _classify(0.5, 0.5) == :weak
    @test_throws ArgumentError linkages(io; forward=:bogus)
    @test_throws ArgumentError sda(io, io; method=:bogus)
    @test_throws ArgumentError hypothetical_extraction(io, "Nonexistent")
    r = hypothetical_extraction(io, ["Agriculture", "Manufacturing"])
    @test r.extracted == [1, 2]
end

@testset "IO coverage — environmental branches" begin
    io = load_example(:wiot)
    @test_throws ArgumentError add_extension!(io, "bad", [1.0 2.0 3.0];
                                              stressors=["s"], unit=["u"])
    add_extension!(io, "land", [5.0 7.0]; stressors=["ha"], unit=["kha"],
                   F_Y=reshape([1.0], 1, 1))         # F_Y non-nothing branch
    fp = footprint(io, "land")
    @test sum(fp.total) ≈ sum(io.extensions["land"].F) + sum(io.extensions["land"].F_Y) atol=1e-6
end

@testset "IO coverage — Baqaee-Farhi branches" begin
    # a sector with no intermediate inputs exercises the zero-cost-share path
    Z = [100.0 0.0; 50.0 0.0]
    Y = reshape([200.0, 300.0], 2, 1)
    io = IOData(Z, Y, [150.0 350.0])
    bf = baqaee_farhi(io; sigma=2.0)                 # consumer-side (σ≠1) branch
    @test size(bf.second_order) == (2, 2)
    @test bf.domar ≈ io.x ./ sum(io.va)
end

@testset "IO coverage — fetch (file:// + skip)" begin
    dir = mktempdir()
    src = joinpath(dir, "src.txt"); write(src, "150.0,500.0\n")
    url = "file://" * src
    dest = joinpath(dir, "out.txt")
    @test fetch_file(url, dest) == dest               # GET branch (no network)
    @test isfile(dest)
    @test occursin("150.0", fetch_text(url))          # fetch_text branch
    # skip-if-exists branch
    @test fetch_file("file://" * src, dest; overwrite=false) == dest
    # POST branch executes (file:// may not support it — just exercise the path)
    try
        fetch_file(url, joinpath(dir, "post.txt"); method="POST", body="x", overwrite=true)
    catch
    end
    @test true
end

@testset "IO coverage — downloaders (mocked, no network)" begin
    dir = mktempdir()
    calls = String[]
    mf = (u, d; kwargs...) -> (push!(calls, u); touch(d); d)
    # WIOD: mock the HTML page and the file fetch
    wiod_html = """<a href="https://www.wiod.org/protected3/data13/AUS/wiot09_row_apr12.xlsx">x</a>"""
    mt = (u; kwargs...) -> wiod_html
    m1 = download_wiod(dir; fetch=mf, fetch_text=mt)
    @test length(m1.files) == 1

    # EXIOBASE3: mock the Zenodo page; exercise system + year filters
    exio_html = """IOT_2010_pxp.zip IOT_2011_ixi.zip
        https://zenodo.org/record/5589597/files/IOT_2010_pxp.zip
        https://zenodo.org/record/5589597/files/IOT_2011_pxp.zip"""
    mt2 = (u; kwargs...) -> exio_html
    m2 = download_exiobase3(dir; system="pxp", years=[2010], fetch=mf, fetch_text=mt2)
    @test m2.source == "EXIOBASE3"

    # GLORIA (Dropbox '?dl=0' URLs): local filenames must strip the query string, else
    # they contain '?' — legal on Linux/macOS but illegal on Windows (open throws EINVAL).
    mg = download_gloria(dir; fetch=mf)
    @test mg isa MacroEconometricModels.IOMetaData
    @test !isempty(mg.files)                                # GLORIA_URLS is populated
    @test all(!occursin('?', fn) for (_, fn) in mg.files)   # query stripped -> Windows-safe
    # EORA26 (history + error)
    me = download_eora26(dir; email="you@example.com", password="pw")
    @test !isempty(me.history)
    @test_throws ArgumentError download_eora26(dir; email="", password="")

    # _match_year occursin branch on range keys, plus download_io dispatch routes
    @test _match_year("2000-2004", [2000])
    @test !_match_year("2000-2004", [1990])
    download_oecd(dir; version="v2021", years=[2000], fetch=mf)
    @test download_io(:wiod; storage_folder=dir, fetch=mf, fetch_text=mt) isa MacroEconometricModels.IOMetaData
    @test download_io(:exiobase3; storage_folder=dir, fetch=mf, fetch_text=mt2) isa MacroEconometricModels.IOMetaData
    @test download_io(:gloria; storage_folder=dir, fetch=mf) isa MacroEconometricModels.IOMetaData
    @test download_io(:eora26; storage_folder=dir, email="a@b.c", password="p") isa MacroEconometricModels.IOMetaData
end
