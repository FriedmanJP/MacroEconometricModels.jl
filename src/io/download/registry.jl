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
              versions=["053"],
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

# --- Integrity verification (#250) -------------------------------------------

"""
    IO_CHECKSUMS :: Dict{String,String}

Registry of expected SHA-256 digests (lowercase hex) for downloadable MRIO
archives, keyed by their exact download URL. **Empty by default** — real digests
require a maintainer to fetch each archive once and record its hash. A URL that is
absent from this registry is still downloaded, but its integrity is NOT verified
and a warning is emitted (see the `verify` keyword of [`download_io`](@ref)).

Populate an entry with [`io_file_digest`](@ref):

```julia
MacroEconometricModels.IO_CHECKSUMS[url] = io_file_digest("downloaded_archive.zip")
```
"""
const IO_CHECKSUMS = Dict{String,String}()

"""
    io_file_digest(path) -> String

Return the SHA-256 digest (lowercase hex) of the file at `path`. Maintainer helper
for populating [`IO_CHECKSUMS`](@ref) after fetching an archive once.
"""
io_file_digest(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))

"""
    _verify_download(url, path; registry=IO_CHECKSUMS) -> path

Check the downloaded file at `path` against the expected SHA-256 for `url`:

- registered digest **matches** → return `path`;
- registered digest **mismatches** → throw an error naming `url` with expected
  vs. actual (the archive is corrupt or has been substituted);
- **no** registered digest → emit a warning and return `path` (download allowed
  but unverified — the honest default while [`IO_CHECKSUMS`](@ref) is unpopulated).
"""
function _verify_download(url::AbstractString, path::AbstractString; registry=IO_CHECKSUMS)
    expected = get(registry, String(url), nothing)
    if expected === nothing
        @warn "No registered SHA-256 checksum for $url; the downloaded file is NOT " *
              "integrity-verified. Populate MacroEconometricModels.IO_CHECKSUMS to enable verification."
        return path
    end
    actual = io_file_digest(path)
    actual == expected || throw(ErrorException(
        "SHA-256 checksum mismatch for $url\n" *
        "  expected: $expected\n" *
        "  actual:   $actual\n" *
        "The downloaded file is corrupt or has been substituted; refusing to use it."))
    path
end
