using Test, MacroEconometricModels
import SHA
using MacroEconometricModels: download_oecd, download_io, OECD_URLS, IOMetaData, IO_CHECKSUMS

@testset "download_oecd with mock fetcher" begin
    # verify=false on the pure-mechanics tests (URLs are unregistered → would only warn)
    calls = String[]
    mockfetch = (url, dest; kwargs...) -> (push!(calls, url); touch(dest); dest)
    dir = mktempdir()
    meta = download_oecd(dir; version="v2023", years=nothing, fetch=mockfetch, verify=false)
    @test !isempty(calls)                          # attempted downloads
    @test length(meta.files) == length(calls)
    @test all(occursin("stats.oecd.org", u) || occursin("oecd.org", u) for u in calls)
    @test all(startswith(u, "https://") for u in calls)   # #250: HTTPS everywhere

    # year filtering reduces the set (v2016 has per-year keys)
    calls2 = String[]
    mf2 = (url, dest; kwargs...) -> (push!(calls2, url); touch(dest); dest)
    download_oecd(dir; version="v2016", years=[1995, 1996], fetch=mf2, verify=false)
    @test length(calls2) == 2

    # unknown version errors
    @test_throws ArgumentError download_oecd(dir; version="v9999", fetch=mockfetch, verify=false)

    # dispatch wrapper routes by symbol
    @test download_io(:oecd; storage_folder=dir, version="v2023",
                      fetch=mockfetch, verify=false) isa IOMetaData
    @test_throws ArgumentError download_io(:nonsense; storage_folder=dir)
end

@testset "download verify=true integrity check (#250)" begin
    url = OECD_URLS["v2016"]["1995"]                # a single, real https OECD URL
    @test startswith(url, "https://")

    # Register the digest of the AUTHENTIC payload, then have the fetcher write a
    # TAMPERED payload → the post-download SHA-256 check must throw, naming the URL.
    authentic = b"authentic ICIO 2016 table bytes"
    IO_CHECKSUMS[url] = bytes2hex(SHA.sha256(authentic))
    try
        tamper = (u, dest; kwargs...) -> (write(dest, b"tampered payload"); dest)
        err = @test_throws ErrorException download_oecd(mktempdir();
                    version="v2016", years=[1995], fetch=tamper, verify=true)
        @test occursin(url, err.value.msg)          # error names the source URL
        @test occursin("mismatch", lowercase(err.value.msg))

        # A matching payload passes verification and returns the log.
        honest = (u, dest; kwargs...) -> (write(dest, authentic); dest)
        @test download_oecd(mktempdir(); version="v2016", years=[1995],
                            fetch=honest, verify=true) isa IOMetaData

        # verify=false bypasses the check entirely (tampered bytes accepted).
        @test download_oecd(mktempdir(); version="v2016", years=[1995],
                            fetch=tamper, verify=false) isa IOMetaData
    finally
        delete!(IO_CHECKSUMS, url)
    end
end
