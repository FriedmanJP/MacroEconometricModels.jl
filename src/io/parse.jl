# parse.jl — turn downloaded files into IOData (zip/xlsx via package extensions)

"""
    parse_io(path; source, year=nothing, kwargs...) -> IOData

Parse a downloaded IO file into an [`IOData`](@ref). Dispatches on the file
extension: `.csv`/`.tsv`/`.txt` are parsed in-core (via `DelimitedFiles`), while
`.zip` and `.xlsx` require the optional `ZipFile` / `XLSX` packages (loaded as
package extensions) and raise an actionable error if those are not available.
"""
function parse_io(path::AbstractString; source::Symbol, year=nothing, kwargs...)
    ext = lowercase(splitext(path)[2])
    # Always forward provenance so the returned IOData carries source/year (#519).
    src = String(source)
    if ext in (".csv", ".tsv", ".txt")
        return _parse_csv_io(path; source=src, year=year, kwargs...)
    elseif ext == ".zip"
        return _parse_zip_io(path; source=src, year=year, kwargs...)
    elseif ext in (".xlsx", ".xls")
        return _parse_xlsx_io(path; source=src, year=year, kwargs...)
    else
        throw(ArgumentError("unsupported file type '$ext' for parse_io"))
    end
end

"""
    _parse_csv_io(path; n_sectors, n_fd=1, sectors=String[], delim=',',
                  source="", year=nothing) -> IOData

Parse a delimited IO block: the first `n_sectors` columns are the intermediate
matrix `Z`, the next `n_fd` columns are final demand. Tables are built with
`check=false` and a single derived value-added row from the column balance.
"""
function _parse_csv_io(path::AbstractString; n_sectors::Int, n_fd::Int=1,
                       sectors=String[], delim::AbstractChar=',',
                       source::AbstractString="", year=nothing)
    raw = readdlm(path, delim, Float64)
    Z = raw[1:n_sectors, 1:n_sectors]
    Y = raw[1:n_sectors, n_sectors+1:n_sectors+n_fd]
    IOData(Z, Y, vec(sum(Z, dims=2)) .+ vec(sum(Y, dims=2));
           sectors=sectors, source=String(source), year=year, check=false)
end

# Extension entry points — real methods live in ext/ and override these.
_parse_zip_io(path; kwargs...) =
    error("Parsing zipped IO archives requires the ZipFile package. " *
          "Run `]add ZipFile` and `using ZipFile` to enable it.")
_parse_xlsx_io(path; kwargs...) =
    error("Parsing Excel IO tables requires the XLSX package. " *
          "Run `]add XLSX` and `using XLSX` to enable it.")
