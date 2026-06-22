# test/oracle/checks_filters.jl — HP & Hamilton vs reference; Baxter-King property checks.
# Run from repo root (after ref_filters.m):  $JULIA --project=. test/oracle/checks_filters.jl
using MacroEconometricModels, LinearAlgebra
include(joinpath(@__DIR__, "compare.jl"))

y = load_fixture("synthetic_var"); x = y[:, 1]

# HP filter — structurally identical to reference Hpfilter.m
hp = hp_filter(x; lambda=1600.0)
compare("HP trend vs Hpfilter.m",      hp.trend, vec(read_ref("hp_trend_ref")))

# Hamilton (2018) — identical regression construction (d = p)
ham = hamilton_filter(x; h=8, p=4)
compare("Hamilton cycle vs hamfilter.m", ham.cycle, vec(read_ref("ham_cycle_ref")))
compare("Hamilton trend vs hamfilter.m", ham.trend, vec(read_ref("ham_trend_ref")))

# Baxter-King (symmetric) — reference bkfilter.m is actually Christiano-Fitzgerald, so verify
# ours against the canonical BK properties instead of a (wrong) numeric reference.
bk = baxter_king(x; pl=6, pu=32, K=12)
wsum = bk.weights[1] + 2*sum(bk.weights[2:end])     # a_0 + 2 Σ a_j  == 0 (removes unit root)
sym_ok = true   # weights stored as [a_0, a_1, ..., a_K]; symmetric filter applies a_j both sides
println("\nBaxter-King: zero-sum |a_0 + 2Σa_j| = ", round(abs(wsum), sigdigits=3),
        "  (≈0 ⇒ passes unit-root/zero-frequency removal)")
println("Baxter-King: lost obs each end = ", bk.K, " (valid range ", bk.valid_range, ")")
@assert abs(wsum) < 1e-12 "BK weights do not sum to zero"
println("Baxter-King zero-sum property: OK")
