using Test, MacroEconometricModels
import SHA
using MacroEconometricModels: scrape_links, _log_download!, IO_HEADERS, IOMetaData,
                              _verify_download, fetch_file

@testset "scrape_links & download log (no network)" begin
    html = """<a href="http://x/IOT_2010_pxp.zip">a</a>
              <a href="http://x/IOT_2011_pxp.zip">b</a> other"""
    urls = scrape_links(html, r"http://\S+?IOT_\d{4}_pxp\.zip")
    @test urls == ["http://x/IOT_2010_pxp.zip", "http://x/IOT_2011_pxp.zip"]

    @test haskey(IO_HEADERS, "User-Agent")
    # G-16 (#254): honest User-Agent, not a spoofed browser string.
    ua = IO_HEADERS["User-Agent"]
    @test !occursin("Mozilla", ua) && !occursin("Firefox", ua)
    @test occursin("MacroEconometricModels", ua)

    meta = IOMetaData(; source="test")
    _log_download!(meta, "http://x/file.zip", "file.zip")
    @test meta.files == ["http://x/file.zip" => "file.zip"]
    @test length(meta.history) == 1
    @test occursin("file.zip", meta.history[1])
end

@testset "SHA-256 integrity verification (#250)" begin
    dir = mktempdir()
    payload = rand(UInt8, 512)
    path = joinpath(dir, "archive.zip"); write(path, payload)

    # io_file_digest is the exact SHA-256 of the bytes
    digest = io_file_digest(path)
    @test digest == bytes2hex(SHA.sha256(payload))
    @test length(digest) == 64

    url = "https://example.org/archive.zip"

    @testset "matching digest passes" begin
        reg = Dict(url => digest)
        @test _verify_download(url, path; registry=reg) == path   # no throw
    end

    @testset "mismatching digest throws, naming the URL" begin
        reg = Dict(url => "0"^64)                                  # wrong digest
        err = @test_throws ErrorException _verify_download(url, path; registry=reg)
        @test occursin(url, err.value.msg)
        @test occursin("expected", err.value.msg) && occursin("actual", err.value.msg)
    end

    @testset "unregistered URL warns and passes (unverified)" begin
        @test_logs (:warn,) _verify_download(url, path; registry=Dict{String,String}())
        @test _verify_download(url, path; registry=Dict{String,String}()) == path
    end

    @testset "file:// fetch_file preserves bytes → digest round-trips" begin
        # Downloads.download handles file:// via libcurl for real GET coverage.
        dest = joinpath(mktempdir(), "out.zip")
        fetch_file("file://" * path, dest)
        @test read(dest) == payload
        @test io_file_digest(dest) == digest
        # verified download of the file:// fixture against its true digest
        @test _verify_download("file://" * path, dest;
                               registry=Dict("file://" * path => digest)) == dest
    end
end
