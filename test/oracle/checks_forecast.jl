# test/oracle/checks_forecast.jl — unconditional VAR forecast recursion vs forecasts.m.
# Run from repo root (after ref_forecast.m):  $JULIA --project=. test/oracle/checks_forecast.jl
using MacroEconometricModels, LinearAlgebra
include(joinpath(@__DIR__, "compare.jl"))

y = load_fixture("synthetic_var"); n, p = 3, 2
m = estimate_var(y, p; check_stability=false)
fhor = 12
ours = predict(m, fhor)                       # h × n, deterministic recursion
compare("uncond forecast vs forecasts.m", ours, read_ref("fcast_noshock_ref"))
