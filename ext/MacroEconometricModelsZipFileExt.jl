module MacroEconometricModelsZipFileExt

using MacroEconometricModels, ZipFile
import MacroEconometricModels: _parse_zip_io, IOData

# Parse a zipped delimited IO block. `member` selects the file inside the
# archive (defaults to the first); the first `n_sectors` columns are Z and the
# next `n_fd` are final demand.
function _parse_zip_io(path::AbstractString; source=nothing, year=nothing,
                       member::AbstractString="", n_sectors::Int=0, n_fd::Int=1,
                       sectors=String[], delim::AbstractChar=',', kwargs...)
    r = ZipFile.Reader(path)
    try
        target = if isempty(member)
            first(r.files)
        else
            idx = findfirst(f -> f.name == member, r.files)
            idx === nothing && throw(ArgumentError("member '$member' not found in $path"))
            r.files[idx]
        end
        content = read(target, String)
        rows = Vector{Vector{Float64}}()
        for line in split(strip(content), '\n')
            s = strip(line)
            isempty(s) && continue
            push!(rows, parse.(Float64, split(s, delim)))
        end
        M = reduce(vcat, (reshape(row, 1, :) for row in rows))
        ns = n_sectors == 0 ? size(M, 1) : n_sectors
        Z = M[1:ns, 1:ns]
        Y = M[1:ns, ns+1:ns+n_fd]
        return IOData(Z, Y, vec(sum(Z, dims=2)) .+ vec(sum(Y, dims=2));
                      sectors=sectors, check=false)
    finally
        close(r)
    end
end

end # module
