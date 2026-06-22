# registry.jl — registry of downloadable MRIO sources (pymrio-derived catalog)

"Catalog of downloadable IO/MRIO sources. Per-version URL tables live in `sources.jl`."
const IO_SOURCES = Dict{Symbol,NamedTuple}(
    :oecd => (name="OECD ICIO", needs_credentials=false,
              versions=["v2016", "v2018", "v2021", "v2023"],
              note="Inter-Country Input-Output tables; fixed per-version URLs."),
    :wiod => (name="WIOD 2013", needs_credentials=false,
              versions=["2013"],
              note="World Input-Output Database; HTML scrape for wiot*.xlsx."),
    :exiobase3 => (name="EXIOBASE 3", needs_credentials=false,
              versions=["3.8.2"],
              note="Zenodo-hosted; record page scraped for IOT_YYYY_{pxp,ixi}.zip."),
    :eora26 => (name="EORA26", needs_credentials=true,
              versions=["26"],
              note="Requires worldmrio.com account (email/password)."),
    :gloria => (name="GLORIA", needs_credentials=false,
              versions=["057"],
              note="Global Resource Input-Output Assessment; fixed URL set."),
)

"Printable table of available IO/MRIO sources, returned by [`list_io_sources`](@ref)."
struct IOSourceTable
    rows::Vector{Tuple{Symbol,NamedTuple}}
end

"""
    list_io_sources() -> IOSourceTable

List the IO/MRIO sources known to [`download_io`](@ref), with versions and
whether each requires credentials.
"""
list_io_sources() = IOSourceTable([(k, IO_SOURCES[k]) for k in sort(collect(keys(IO_SOURCES)))])

function Base.show(io::IO, t::IOSourceTable)
    println(io, "Available IO/MRIO sources:")
    for (k, v) in t.rows
        cred = v.needs_credentials ? " (credentials required)" : ""
        println(io, "  :$k — $(v.name)$cred")
        println(io, "      versions: $(join(v.versions, ", "))")
        println(io, "      $(v.note)")
    end
end
