using Test, MacroEconometricModels
using MacroEconometricModels: IO_SOURCES, list_io_sources

@testset "Source registry" begin
    for s in (:oecd, :wiod, :exiobase3, :eora26, :gloria)
        @test haskey(IO_SOURCES, s)
        @test IO_SOURCES[s].name isa String
    end
    @test IO_SOURCES[:eora26].needs_credentials == true
    @test IO_SOURCES[:oecd].needs_credentials == false
    t = list_io_sources()
    buf = IOBuffer(); show(buf, t)
    out = String(take!(buf))
    @test occursin("OECD", out)
    @test occursin("EORA", out) || occursin("Eora", out)
end
