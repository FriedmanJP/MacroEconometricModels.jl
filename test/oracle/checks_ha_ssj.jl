# test/oracle/checks_ha_ssj.jl — HA / sequence-space-Jacobian (SSJ) cross-check.
#
# Primary oracle: the Python `sequence-jacobian` toolkit (Auclert, Bardóczy,
# Rognlie & Straub 2021). That package is NOT importable in this read-only
# environment, so its reference values are produced OFFLINE by
# `python/gen_ha_ssj_reference.py` and committed as CSV fixtures under
# `ha_ssj_ref/`. When those fixtures are present this harness diffs our household
# Jacobians / GE IRFs against them via `compare()`; otherwise it runs the in-env
# consistency checks alone (which already catch the pre-#226 anticipation-zeroing
# bug and the pre-#227 observation-map bug).
#
# Manual / weekly harness — NOT part of the default `runtests.jl` groups. Guarded
# by `MACRO_ORACLE_TESTS`. Run from the repository root:
#     MACRO_ORACLE_TESTS=1 $JULIA --project=. test/oracle/checks_ha_ssj.jl
#
# Depends on: #226 (real fake-news Jacobian), #227/#228 (Ho-Kalman C/D obs map),
# #229/#230 (KS/Reiter GE), #231 (income normalization), #232/#233/#234 — i.e. the
# whole Stage-9 HA rebuild. Feeds T158 (#257, the extended oracle harness).

using MacroEconometricModels, LinearAlgebra, DelimitedFiles
const MEM = MacroEconometricModels
include(joinpath(@__DIR__, "compare.jl"))

const HA_REF_DIR = joinpath(@__DIR__, "ha_ssj_ref")

if !haskey(ENV, "MACRO_ORACLE_TESTS")
    @info "checks_ha_ssj.jl is a manual oracle harness; set MACRO_ORACLE_TESTS=1 to run it."
    exit(0)
end

"Read a committed HA reference CSV, or `nothing` if the offline fixtures are absent."
function read_ha_ref(name::AbstractString)
    path = joinpath(HA_REF_DIR, name * ".csv")
    isfile(path) ? readdlm(path, ',', Float64) : nothing
end

# ── In-env consistency checks (no Python needed) ─────────────────────────────

"""
Anticipation: the fake-news SSJ Jacobian `J[t,s] = ∂O_t/∂I_s` must be DENSE, with
non-zero anticipation entries `J[t,s] ≠ 0` for `t < s` (households respond before
an announced future price change). The pre-#226 brute force zeroed the `t<s`
upper triangle, so this alone discriminates the fix.
"""
function check_anticipation(label, J::AbstractMatrix)
    Th = size(J, 1)
    up = maximum(abs(J[t, s]) for t in 1:Th, s in 1:Th if t < s; init=0.0)
    lo = maximum(abs(J[t, s]) for t in 1:Th, s in 1:Th if t >= s; init=0.0)
    pass = up > 1e-6 * max(lo, 1.0)
    println(rpad("$label anticipation J[t<s]≠0", 40), pass ? "PASS" : "FAIL",
            "  upper(t<s)=", round(up, sigdigits=3), "  lower(t≥s)=", round(lo, sigdigits=3))
    return pass
end

"""
Ho-Kalman realization consistency (#227): the minimal state-space realization of a
Jacobian column reproduces that column, `h[1]=D`, `h[k]=C·A^{k-2}·B` for `k≥2`.
"""
function check_hokalman(label, col::AbstractVector)
    Th = length(col)
    irf_seq = [reshape([col[t]], 1, 1) for t in 1:Th]
    k = max(min(20, div(Th, 2) - 1), 1)
    G1, impact, _, _, _, C, D = MEM._ho_kalman(irf_seq, 1, 1, k)
    recon = zeros(Th)
    recon[1] = D[1, 1]
    Ah = Matrix{Float64}(I, size(G1)...)
    for h in 2:Th
        recon[h] = (C * (Ah * impact))[1, 1]
        Ah = Ah * G1
    end
    return compare("$label Ho-Kalman realization", recon, col; rtol=1e-4, atol=1e-5)
end

function run_checks()
    all_pass = true

    # ── Krusell–Smith ────────────────────────────────────────────────────────
    println("── Krusell–Smith (SSJ) ──")
    ks = load_ha_example(:krusell_smith)
    ss_ks = compute_steady_state(ks; max_iter=150, tol=1e-6)
    println(rpad("KS steady-state market clearing", 40),
            abs(ss_ks.excess_demand) < 1e-3 ? "PASS" : "FAIL",
            "  |excess|=", round(abs(ss_ks.excess_demand), sigdigits=3))
    all_pass &= abs(ss_ks.excess_demand) < 1e-3

    Th = 200
    J_r_K = MEM._ssj_jacobian(ss_ks, ks.individual, ks.grid, ks.income, :r, :K; T_horizon=Th)
    J_r_A = MEM._ssj_jacobian(ss_ks, ks.individual, ks.grid, ks.income, :r, :A; T_horizon=Th)
    all_pass &= check_anticipation("KS dK/dr", J_r_K)
    all_pass &= check_hokalman("KS dK/dr col-1", J_r_K[:, 1]).pass
    # :K and :A both aggregate assets (#240/H-16), so the columns must coincide.
    println(rpad("KS dK/dr == dA/dr (H-16)", 40),
            maximum(abs.(J_r_K .- J_r_A)) < 1e-10 ? "PASS" : "FAIL")
    all_pass &= maximum(abs.(J_r_K .- J_r_A)) < 1e-10

    # ── One-asset HANK ───────────────────────────────────────────────────────
    println("\n── One-asset HANK (SSJ) ──")
    hank = MEM._one_asset_hank_example()
    ss_h = compute_steady_state(hank; max_iter=150, tol=1e-6)
    J_hank = MEM._ssj_jacobian(ss_h, hank.individual, hank.grid, hank.income, :r, :K; T_horizon=Th)
    all_pass &= check_anticipation("HANK dK/dr", J_hank)
    all_pass &= check_hokalman("HANK dK/dr col-1", J_hank[:, 1]).pass

    # ── Huggett GE-clearing identity ─────────────────────────────────────────
    println("\n── Huggett (SSJ GE clearing) ──")
    hg = load_ha_example(:huggett)
    ss_hg = compute_steady_state(hg; max_iter=150, tol=1e-6)
    sol_hg = solve(hg; method=:ssj, ss=ss_hg, T_horizon=Th, n_reduced=20)
    H_U = sol_hg.jacobians[:H_U]; H_Z = sol_hg.jacobians[:H_Z]
    rho = get(hg.het_params, :rho_e, 0.9)
    dw = [rho^(t - 1) for t in 1:Th]
    dr = -(H_U \ (H_Z * dw))                       # market-clearing rate path
    # The Ho-Kalman observation map must reproduce the clearing rate on impact.
    println(rpad("Huggett dr[1] < 0 (endowment↑ ⇒ r↓)", 40), dr[1] < 0 ? "PASS" : "FAIL",
            "  dr[1]=", round(dr[1], sigdigits=3))
    all_pass &= dr[1] < 0
    ge = compare("Huggett D_obs == dr[1]", [sol_hg.D_obs[1, 1]], [dr[1]]; rtol=1e-6, atol=1e-8)
    all_pass &= ge.pass

    # ── Offline Python sequence-jacobian fixtures (if committed) ──────────────
    println("\n── Python sequence-jacobian fixtures ──")
    ref_ks = read_ha_ref("ks_J_r_A")
    if ref_ks === nothing
        @info "No offline sequence-jacobian fixtures in $HA_REF_DIR — run " *
              "python/gen_ha_ssj_reference.py offline and commit its CSVs to enable " *
              "the external cross-check. In-env consistency checks above already ran."
    else
        all_pass &= compare("KS dA/dr vs sequence-jacobian", J_r_A, ref_ks; rtol=1e-3, atol=1e-4).pass
        ref_hank = read_ha_ref("hank_J_r_A")
        ref_hank === nothing || (all_pass &= compare("HANK dK/dr vs sequence-jacobian",
                                                     J_hank, ref_hank; rtol=1e-3, atol=1e-4).pass)
    end

    println("\n", all_pass ? "ALL HA/SSJ CHECKS PASSED" : "SOME HA/SSJ CHECKS FAILED")
    return all_pass
end

run_checks()
