# linkages.jl — backward/forward linkages, Rasmussen indices, key sectors

"Linkage indices and key-sector classification for an IO table."
struct LinkageResult
    backward::Vector{Float64}
    forward::Vector{Float64}
    Ui::Vector{Float64}            # power of dispersion (backward, normalized)
    Uj::Vector{Float64}            # sensitivity of dispersion (forward, normalized)
    classification::Vector{Symbol}
    sectors::Vector{String}
end

"""
    linkages(io; forward=:ghosh) -> LinkageResult

Backward linkages (column sums of `L`) and forward linkages (row sums of the
Ghosh inverse `G` by default, or of `L` with `forward=:leontief`). Rasmussen
power-of-dispersion (`Ui`) and sensitivity-of-dispersion (`Uj`) indices are the
linkages normalized by their overall average, and the per-sector
classification follows the `(Ui, Uj)` quadrants.
"""
function linkages(io::IOData; forward::Symbol=:ghosh)
    L = leontief_inverse(io)
    n = size(L, 1)
    backward = vec(sum(L, dims=1))                      # column sums of L
    fwd = if forward == :ghosh
        vec(sum(ghosh_inverse(io), dims=2))             # row sums of G
    elseif forward == :leontief
        vec(sum(L, dims=2))                             # row sums of L (Chenery-Watanabe)
    else
        throw(ArgumentError("forward must be :ghosh or :leontief"))
    end
    Ui = backward ./ (sum(backward) / n)
    Uj = fwd ./ (sum(fwd) / n)
    classification = [_classify(Ui[i], Uj[i]) for i in 1:n]
    LinkageResult(backward, fwd, Ui, Uj, classification, copy(io.sectors))
end

_classify(ui, uj) = ui > 1 && uj > 1 ? :key :
                    ui > 1 && uj <= 1 ? :backward :
                    ui <= 1 && uj > 1 ? :forward : :weak

"Alias for [`linkages`](@ref) returning the Rasmussen indices."
rasmussen(io::IOData) = linkages(io)
"Per-sector key-sector classification (`:key`/`:forward`/`:backward`/`:weak`)."
key_sectors(io::IOData) = linkages(io).classification
