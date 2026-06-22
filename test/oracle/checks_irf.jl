# test/oracle/checks_irf.jl — Cholesky IRF, FEVD, structural shocks, HD identity vs reference.
# Run from repo root (after ref_irf_fevd.m):  $JULIA --project=. test/oracle/checks_irf.jl
using MacroEconometricModels, LinearAlgebra, Random
const MEM = MacroEconometricModels
include(joinpath(@__DIR__, "compare.jl"))

y = load_fixture("synthetic_var"); n, p = 3, 2
m = estimate_var(y, p; check_stability=false)

dims = Int.(vec(read_ref("irf_dims")))        # [n hor n]
hor = dims[2]
ir_ref = reshape(vec(read_ref("irf_ref")), dims[1], dims[2], dims[3])   # [var, h, shock]

# Sanity: the Sigma/alpha Octave used must equal ours (same model).
compare("Sigma fed to both", m.Sigma, read_ref("irf_Sigma"))
compare("alpha fed to both", m.B[2:end, :], read_ref("irf_alpha"))

# Our Cholesky IRF (Q = I) -> (hor, n, n); permute to (var, h, shock) to match the reference.
IRF = MEM.compute_irf(m, Matrix{Float64}(I, n, n), hor)
IRF_p = permutedims(IRF, (2, 1, 3))
compare("Cholesky IRF vs iresponse", IRF_p, ir_ref)

# Our cumulative FEVD shares at horizon `hor` (×100) vs reference fevd.m.
_, props = MEM._compute_fevd(IRF, n, hor)
compare("FEVD vs fevd.m (horizon $hor)", props[:, :, hor] .* 100, read_ref("fevd_ref"))

# Historical decomposition: identity holds exactly, and shocks match ε = L^{-1} u.
hd = MEM.historical_decomposition(m, hor; method=:cholesky)
println("\nHD identity (contrib + initial == actual): ", MEM.verify_decomposition(hd))
L = MEM.safe_cholesky(m.Sigma)
eps_direct = (inv(Matrix(L)) * m.U')'         # ε_t = L^{-1} u_t  (Cholesky, Q=I)
compare("HD structural shocks ε = L⁻¹u", hd.shocks, eps_direct)

# Long-run (Blanchard-Quah): compare SQUARED IRFs (sign-invariant; ours omits the Q(1,1)>0
# normalization the reference applies, so per-shock signs may differ — see F-05).
Q_lr = MEM.identify_long_run(m)
IRF_lr = permutedims(MEM.compute_irf(m, Q_lr, hor), (2, 1, 3))   # [var, h, shock]
ir_lr_ref = reshape(vec(read_ref("irf_lr_ref")), dims[1], dims[2], dims[3])
compare("long-run BQ IRF² (sign-free)", IRF_lr .^ 2, ir_lr_ref .^ 2)
# Defining BQ property: long-run cumulative impact matrix is lower-triangular.
LR_cum = dropdims(sum(IRF_lr; dims=2); dims=2)        # [var, shock] long-run cumulative
upper_off = maximum(abs.(LR_cum[i, j] for i in 1:n, j in 1:n if j > i))
println("\nBQ long-run cumulative impact upper-triangle max |.| = ", round(upper_off, sigdigits=3),
        "  (≈0 ⇒ lower-triangular, permanent shock ordered first)")
