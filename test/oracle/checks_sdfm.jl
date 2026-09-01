# test/oracle/checks_sdfm.jl — FGLR Cholesky fixture vs estimate_structural_dfm.
# Run from repo root:  julia --project=. test/oracle/checks_sdfm.jl
using MacroEconometricModels, LinearAlgebra, DelimitedFiles
include(joinpath(@__DIR__, "compare.jl"))

const SDFM_REF = joinpath(@__DIR__, "sdfm_ref")
X = readdlm(joinpath(SDFM_REF, "X.csv"), ',', Float64)
Kref = readdlm(joinpath(SDFM_REF, "K.csv"), ',', Float64)
B0ref = readdlm(joinpath(SDFM_REF, "B0.csv"), ',', Float64)
irfref = readdlm(joinpath(SDFM_REF, "irf.csv"), ',', Float64)

sdfm = estimate_structural_dfm(X, 2; r=2, p=1, H=12, identification=:cholesky,
    order=[1, 2], standardize=true, method=:fglr)
rK = compare("FGLR K", sdfm.K, Kref; rtol=1e-6, atol=1e-6)
rB = compare("FGLR impact B0", sdfm.B0, B0ref; rtol=1e-6, atol=1e-6)
ir = irf(sdfm, 12).values
H, N, q = size(ir)
rI = compare("FGLR 12-step IRF", reshape(ir, H * N, q), irfref; rtol=1e-6, atol=1e-6)
maxabs = max(rK.maxabs, rB.maxabs, rI.maxabs)
maxrel = max(rK.maxrel, rB.maxrel, rI.maxrel)
println("max abs/rel = ", maxabs, " / ", maxrel)
(rK.pass && rB.pass && rI.pass) || error("SDFM fixture comparison failed")
