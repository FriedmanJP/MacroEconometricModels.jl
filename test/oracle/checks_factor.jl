# test/oracle/checks_factor.jl — factor PCA + Bai-Ng selection vs reference baing.m.
# Data is already standardized (by ref_factor.m), so standardize=false on both sides.
# Run from repo root (after ref_factor.m):  $JULIA --project=. test/oracle/checks_factor.jl
using MacroEconometricModels, LinearAlgebra
const MEM = MacroEconometricModels
include(joinpath(@__DIR__, "compare.jl"))

Xs = load_fixture("factor_panel")          # T×N, already standardized
T_obs, N = size(Xs); kmax = 8

# --- PCA common component for r=3 vs reference projection ---
fm = estimate_factors(Xs, 3; standardize=false)
chat_ours = predict(fm)                    # F * Λ'
chat_ref  = read_ref("factor_chat3")
rc = compare("PCA common component (r=3)", chat_ours, chat_ref)
# Also compare to the true projection independently (sign/rotation-free check):
ev = eigen(Symmetric(Xs'Xs / T_obs)); idx = sortperm(ev.values; rev=true)
Vr = ev.vectors[:, idx[1:3]]
proj = Xs * Vr * Vr'                       # U_r U_r' Xs
compare("predict vs true projection X·Vᵣ·Vᵣ'", chat_ours, proj)

# --- Bai-Ng information criteria ---
ic = MEM.ic_criteria(Xs, kmax; standardize=false)
IC_ref = read_ref("factor_IC")             # kmax × 3
compare("Bai-Ng IC1 vs baing jj=1", ic.IC1, IC_ref[:, 1])
compare("Bai-Ng IC2 vs baing jj=2", ic.IC2, IC_ref[:, 2])
compare("Bai-Ng IC3 vs baing jj=3", ic.IC3, IC_ref[:, 3])

# Quantify the IC2 gap: reference adds an extra factor of 2 on log(min(N,T)).
minNT = min(N, T_obs)
extra = [r * (N + T_obs) / (N*T_obs) * log(minNT) for r in 1:kmax]   # one more penalty unit
gap = IC_ref[:, 2] .- ic.IC2
println("\nIC2 gap (ref − ours) vs one extra Bai-Ng penalty term:")
compare("  IC2 gap == extra penalty", gap, extra)
sel_ref = Int.(vec(read_ref("factor_sel")))
println("selected factors  ref(IC1,IC2,IC3) = ", sel_ref,
        "   ours = (", ic.r_IC1, ",", ic.r_IC2, ",", ic.r_IC3, ")")
