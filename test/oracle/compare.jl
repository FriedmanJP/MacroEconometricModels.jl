# test/oracle/compare.jl — comparison primitive reused by every checks_*.jl driver.
using DelimitedFiles
include(joinpath(@__DIR__, "fixtures.jl"))
const ORACLE_OUT = joinpath(@__DIR__, "_out")

"Read a CSV that an Octave ref_*.m script dumped into _out/."
read_ref(name::AbstractString) = readdlm(joinpath(ORACLE_OUT, name * ".csv"), ',', Float64)

"""
    compare(label, ours, theirs; rtol=1e-6, atol=1e-8)

Print a one-line PASS/FAIL verdict and return (pass, maxabs, maxrel). Sizes must match;
align ordering conventions (lags/constant) BEFORE calling.
"""
function compare(label, ours::AbstractArray, theirs::AbstractArray; rtol=1e-6, atol=1e-8)
    size(ours) == size(theirs) || error("$label: size mismatch $(size(ours)) vs $(size(theirs))")
    d = abs.(ours .- theirs)
    maxabs = isempty(d) ? 0.0 : maximum(d)
    denom = max.(abs.(theirs), 1e-12)
    maxrel = isempty(d) ? 0.0 : maximum(d ./ denom)
    pass = all(isapprox.(ours, theirs; rtol=rtol, atol=atol))
    println(rpad(string(label), 40), pass ? "PASS" : "FAIL",
            "  maxabs=", round(maxabs, sigdigits=3), "  maxrel=", round(maxrel, sigdigits=3))
    return (pass=pass, maxabs=maxabs, maxrel=maxrel)
end
