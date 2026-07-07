using Test, MacroEconometricModels
using MacroEconometricModels: download_oecd, download_io, OECD_URLS, IOMetaData

@testset "download_oecd with mock fetcher" begin
    calls = String[]
    mockfetch = (url, dest; kwargs...) -> (push!(calls, url); touch(dest); dest)
    dir = mktempdir()
    meta = download_oecd(dir; version="v2023", years=nothing, fetch=mockfetch)
    @test !isempty(calls)                          # attempted downloads
    @test length(meta.files) == length(calls)
    @test all(occursin("stats.oecd.org", u) || occursin("oecd.org", u) for u in calls)

    # year filtering reduces the set (v2016 has per-year keys)
    calls2 = String[]
    mf2 = (url, dest; kwargs...) -> (push!(calls2, url); touch(dest); dest)
    download_oecd(dir; version="v2016", years=[1995, 1996], fetch=mf2)
    @test length(calls2) == 2

    # unknown version errors
    @test_throws ArgumentError download_oecd(dir; version="v9999", fetch=mockfetch)

    # dispatch wrapper routes by symbol
    @test download_io(:oecd; storage_folder=dir, version="v2023",
                      fetch=mockfetch) isa IOMetaData
    @test_throws ArgumentError download_io(:nonsense; storage_folder=dir)
end
