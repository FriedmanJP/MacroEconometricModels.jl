# fetch.jl — HTTP fetching via the Downloads stdlib (no heavy deps)

"Browser-like headers; plain requests are blocked by several MRIO hosts."
const IO_HEADERS = Dict(
    "User-Agent" =>
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/131.0",
)

"Append a `url => filename` record and a timestamped history line to `meta`."
function _log_download!(meta::IOMetaData, url::AbstractString, filename::AbstractString)
    push!(meta.files, String(url) => String(filename))
    push!(meta.history, "$(Dates.now()): downloaded $(filename) from $(url)")
    meta
end

"Extract all substrings of `html` matching `pattern`."
scrape_links(html::AbstractString, pattern::Regex) =
    String[m.match for m in eachmatch(pattern, html)]

"""
    fetch_file(url, dest; headers, method="GET", body=nothing, overwrite=false) -> dest

Download `url` to `dest` via the `Downloads` stdlib (libcurl: follows redirects,
sends browser headers). Skips the download if `dest` exists and `overwrite` is
false. Supports non-GET methods and a request `body` (e.g. login POST).
"""
function fetch_file(url::AbstractString, dest::AbstractString;
                    headers=IO_HEADERS, method::AbstractString="GET",
                    body=nothing, overwrite::Bool=false)
    (isfile(dest) && !overwrite) && return dest
    mkpath(dirname(dest))
    if method == "GET" && body === nothing
        Downloads.download(url, dest; headers=headers)
    else
        open(dest, "w") do io
            Downloads.request(url; method=method, headers=headers,
                              input=(body === nothing ? nothing : IOBuffer(body)),
                              output=io)
        end
    end
    dest
end

"""
    fetch_text(url; headers=IO_HEADERS) -> String

Fetch `url` and return the response body as a `String` (for HTML scraping).
"""
function fetch_text(url::AbstractString; headers=IO_HEADERS)
    io = IOBuffer()
    Downloads.download(url, io; headers=headers)
    String(take!(io))
end
