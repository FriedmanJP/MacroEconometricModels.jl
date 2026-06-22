module MacroEconometricModelsXLSXExt

using MacroEconometricModels, XLSX
import MacroEconometricModels: _parse_xlsx_io, IOData

# Parse an IO table from an Excel sheet's used range. The first `n_sectors`
# columns are Z and the next `n_fd` are final demand; blanks coalesce to zero.
function _parse_xlsx_io(path::AbstractString; source=nothing, year=nothing,
                        sheet=1, n_sectors::Int=0, n_fd::Int=1,
                        sectors=String[], kwargs...)
    xf = XLSX.readxlsx(path)
    sname = XLSX.sheetnames(xf)[sheet]
    raw = xf[sname][:]                        # used range as a Matrix of values
    M = map(v -> v isa Number ? Float64(v) : 0.0, raw)
    ns = n_sectors == 0 ? size(M, 1) : n_sectors
    Z = M[1:ns, 1:ns]
    Y = M[1:ns, ns+1:ns+n_fd]
    IOData(Z, Y, vec(sum(Z, dims=2)) .+ vec(sum(Y, dims=2)); sectors=sectors, check=false)
end

end # module
