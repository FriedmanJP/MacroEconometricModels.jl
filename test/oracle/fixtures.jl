# test/oracle/fixtures.jl — Julia-side loader for the shared CSV fixtures.
using DelimitedFiles
const ORACLE_DATA = joinpath(@__DIR__, "_data")
load_fixture(name::AbstractString) = readdlm(joinpath(ORACLE_DATA, name * ".csv"), ',', Float64)
