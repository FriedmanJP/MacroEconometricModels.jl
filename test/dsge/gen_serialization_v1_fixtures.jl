# MacroEconometricModels.jl
# Copyright (C) 2025-2026 Wookyung Chung <chung@friedman.jp>
#
# This file is part of MacroEconometricModels.jl.
# Licensed under GPL-3.0-or-later. See LICENSE for details.

# Regenerates test/fixtures/serialization/v1/*.jld2. Run only when
# SERIALIZATION_FORMAT_VERSION bumps.

using MacroEconometricModels

const _DIR = joinpath(@__DIR__, "..", "fixtures", "serialization", "v1")
mkpath(_DIR)

spec = @dsge begin
    parameters: β = 0.99, α = 0.36, δ = 0.025, ρ = 0.9, σ = 0.03
    endogenous: k, c, z
    exogenous: ε
    z[t] = ρ * z[t-1] + σ * ε[t]
    k[t] = exp(z[t]) * k[t-1]^α + (1 - δ) * k[t-1] - c[t]
    1 / c[t] = β * (1 / c[t+1]) * (α * exp(z[t+1]) * k[t]^(α - 1) + 1 - δ)
end
sol = solve(compute_steady_state(spec))
save_model(sol, joinpath(_DIR, "dsge_rbc.jld2"))

hh = only(values(MacroEconometricModels._huggett_example(;
    credit_limit=-2.0, a_max=8.0, n_a=20).agents))
save_model(hh, joinpath(_DIR, "ha_ks.jld2"))

println("wrote ", joinpath(_DIR, "dsge_rbc.jld2"), " and ha_ks.jld2")
