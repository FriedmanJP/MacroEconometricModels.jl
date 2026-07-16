using MacroEconometricModels, Random, LinearAlgebra, Statistics
const _PROJ = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_PROJ, "test", "fixtures.jl"))
include(joinpath(_PROJ, "test", "display", "display_helpers.jl"))

# Exercise a few fixtures; simulate a Windows CRLF golden and confirm the new
# read-side normalization makes it compare equal to the LF-canonical render.
norm_expected(raw) = rstrip(replace(raw, "\r\n" => "\n", "\r" => "\n"), '\n')

allok = true
for f in display_fixtures()
    canon = _canonicalize(_render(f.obj))
    crlf_file = replace(canon, "\n" => "\r\n") * "\r\n"      # what Git-autocrlf checkout yields
    lf_file   = canon * "\n"                                  # normal Linux checkout
    ok_crlf = norm_expected(crlf_file) == canon
    ok_lf   = norm_expected(lf_file)   == canon
    global allok &= (ok_crlf && ok_lf)
    ok_crlf && ok_lf || println("✗ $(f.name): crlf=$ok_crlf lf=$ok_lf")
end
println(allok ? "ALL FIXTURES: CRLF and LF goldens both match ✓" : "SOME FAILED ✗")
