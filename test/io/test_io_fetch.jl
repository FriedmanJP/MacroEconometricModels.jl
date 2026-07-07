using Test, MacroEconometricModels
using MacroEconometricModels: scrape_links, _log_download!, IO_HEADERS, IOMetaData

@testset "scrape_links & download log (no network)" begin
    html = """<a href="http://x/IOT_2010_pxp.zip">a</a>
              <a href="http://x/IOT_2011_pxp.zip">b</a> other"""
    urls = scrape_links(html, r"http://\S+?IOT_\d{4}_pxp\.zip")
    @test urls == ["http://x/IOT_2010_pxp.zip", "http://x/IOT_2011_pxp.zip"]

    @test haskey(IO_HEADERS, "User-Agent")

    meta = IOMetaData(; source="test")
    _log_download!(meta, "http://x/file.zip", "file.zip")
    @test meta.files == ["http://x/file.zip" => "file.zip"]
    @test length(meta.history) == 1
    @test occursin("file.zip", meta.history[1])
end
