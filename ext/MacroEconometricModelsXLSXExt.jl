module MacroEconometricModelsXLSXExt

using MacroEconometricModels, XLSX
import MacroEconometricModels: _parse_xlsx_io, _xlsx_sheet_matrix, IOData

# Parse an IO table from an Excel sheet's used range. The first `n_sectors`
# columns are Z and the next `n_fd` are final demand; blanks coalesce to zero.
function _parse_xlsx_io(path::AbstractString; source=nothing, year=nothing,
                        sheet=1, n_sectors::Int=0, n_fd::Int=1,
                        sectors=String[], kwargs...)
    raw = _xlsx_sheet_matrix(path; sheet=sheet)
    M = map(v -> v isa Number ? Float64(v) : 0.0, raw)
    ns = n_sectors == 0 ? size(M, 1) : n_sectors
    Z = M[1:ns, 1:ns]
    Y = M[1:ns, ns+1:ns+n_fd]
    src = source === nothing ? "" : String(source)
    IOData(Z, Y, vec(sum(Z, dims=2)) .+ vec(sum(Y, dims=2));
           sectors=sectors, source=src, year=year, check=false)
end

function _xlsx_sheet_matrix(path::AbstractString; sheet=1)
    xf = XLSX.readxlsx(path)
    sname = if sheet isa Integer
        XLSX.sheetnames(xf)[Int(sheet)]
    else
        String(sheet)
    end
    # Used range as a Matrix of values (Any / Missing).
    raw = xf[sname][:]
    # Materialise to Array{Any} so callers can mutate cells (WIOD meta blanking).
    A = Array{Any}(undef, size(raw)...)
    for i in eachindex(raw)
        A[i] = raw[i]
    end
    return A
end

end # module
