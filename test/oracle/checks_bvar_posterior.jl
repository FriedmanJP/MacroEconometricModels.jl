# test/oracle/checks_bvar_posterior.jl — BVAR sampler reproduces the analytic NIW posterior.
# Confirms _sample_direct / _draw_inverse_wishart draw from the intended posterior moments.
# Run from repo root:  $JULIA --project=. test/oracle/checks_bvar_posterior.jl
using MacroEconometricModels, LinearAlgebra, Random, Statistics
const MEM = MacroEconometricModels
include(joinpath(@__DIR__, "compare.jl"))

y = load_fixture("synthetic_var"); p, n = 2, 3
Yeff, X = MEM.construct_var_matrices(y, p)
Teff, k = size(Yeff, 1), size(X, 2)

# Reproduce the analytic posterior moments exactly as estimate_bvar does (:normal prior).
κ = 100.0
V0inv = (1/κ) * Matrix{Float64}(I, k, k)
B0 = zeros(k, n); ν0 = n + 2; S0 = Matrix{Float64}(I, n, n)
Vpost = inv(X'X + V0inv); Vpost = 0.5*(Vpost + Vpost')
Bpost = Vpost * (X'Yeff + V0inv*B0)
νpost = ν0 + Teff
Spost = S0 + Yeff'Yeff + B0'V0inv*B0 - Bpost'*(X'X + V0inv)*Bpost
Spost = 0.5*(Spost + Spost')
Sigma_mean_analytic = Spost / (νpost - n - 1)        # IW(scale=Spost, νpost) mean

Random.seed!(20260623)
post = estimate_bvar(y, p; n_draws=40000, sampler=:direct)
B_emp = dropdims(mean(post.B_draws; dims=1); dims=1)         # k×n
S_emp = dropdims(mean(post.Sigma_draws; dims=1); dims=1)     # n×n

r1 = compare("E[B] draws vs analytic B_post",   B_emp, Bpost;               rtol=2e-2, atol=2e-2)
r2 = compare("E[Σ] draws vs Spost/(ν-n-1)",     S_emp, Sigma_mean_analytic; rtol=3e-2, atol=3e-2)

# Gibbs sampler should target the same posterior.
Random.seed!(20260623)
postg = estimate_bvar(y, p; n_draws=40000, sampler=:gibbs, burnin=2000)
Bg = dropdims(mean(postg.B_draws; dims=1); dims=1)
r3 = compare("E[B] gibbs vs analytic B_post",   Bg, Bpost; rtol=3e-2, atol=3e-2)

println("\n  direct & gibbs samplers reproduce the analytic NIW posterior: ",
        all((r1.pass, r2.pass, r3.pass)))
