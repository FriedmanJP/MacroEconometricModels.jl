# test/oracle/checks_bvar_ml.jl — BVAR marginal likelihood vs reference matrictint (F-02).
# Run from repo root (after ref_matrictint.m):  $JULIA --project=. test/oracle/checks_bvar_ml.jl
using MacroEconometricModels, LinearAlgebra
using Distributions: loggamma
const MEM = MacroEconometricModels
include(joinpath(@__DIR__, "compare.jl"))

# Julia reimplementation of the reference matrictint (Sims): log integral of the NIW kernel.
function matrictint(S, df, XXi)
    k = size(XXi, 1); ny = size(S, 1)
    cx = cholesky(Symmetric(XXi)).U
    cs = cholesky(Symmetric(S)).U
    w1 = 0.5*k*ny*log(2π) + ny*sum(log.(diag(cx)))
    lgg = sum(loggamma.(0.5 .* (df .+ collect(0:-1:(1-ny)))))   # ggammaln(ny, df)
    w2 = -df*sum(log.(diag(cs))) + 0.5*df*ny*log(2) + ny*(ny-1)*0.25*log(π) + lgg
    w1 + w2
end
# log multivariate gamma Γ_ny(a)
logmvgamma(ny, a) = ny*(ny-1)*0.25*log(π) + sum(loggamma(a + 0.5*(1 - j)) for j in 1:ny)

# (1) Validate the reimplementation against Octave's matrictint on the fixed case.
S = read_ref("matrictint_S"); XXi = read_ref("matrictint_XXi")
df = Int(read_ref("matrictint_df")[1]); wref = read_ref("matrictint_w")[1]
compare("matrictint reimpl vs octave", [matrictint(S, df, XXi)], [wref])

# (2) Confirm F-02 on the synthetic-fixture BVAR with the default Minnesota prior.
y = load_fixture("synthetic_var"); p, n = 2, 3
hyper = MinnesotaHyperparameters()                       # tau=3, decay=.5, lambda=5, mu=2, omega=2
Yd, Xd = MEM.gen_dummy_obs(y, p, hyper)
Yeff, X = MEM.construct_var_matrices(y, p)
Teff, k = size(Yeff, 1), size(X, 2)
Yaug, Xaug = vcat(Yeff, Yd), vcat(X, Xd)
Kpost, Kprior = Xaug'Xaug, Xd'Xd
Td = size(Yd, 1)
Baug = Kpost \ (Xaug'Yaug); Bprior = Kprior \ (Xd'Yd)
Spost  = (Yaug - Xaug*Baug)' * (Yaug - Xaug*Baug)
Sprior = (Yd   - Xd*Bprior)' * (Yd   - Xd*Bprior)
nu_prior = Td - k; nu_post = Teff + nu_prior

our_ml = MEM.log_marginal_likelihood(y, p, hyper)
ref_ml = matrictint(Spost, nu_post, inv(Kpost)) - matrictint(Sprior, nu_prior, inv(Kprior)) - 0.5*Teff*n*log(2π)
gap_analytic = -0.5*Teff*n*log(π) + (logmvgamma(n, nu_post/2) - logmvgamma(n, nu_prior/2))

println("\n  our log_marginal_likelihood = ", round(our_ml, digits=6))
println("  reference ML (via matrictint)= ", round(ref_ml, digits=6))
println("  ref - our                    = ", round(ref_ml - our_ml, digits=6))
println("  analytic omitted terms       = ", round(gap_analytic, digits=6))
compare("F-02 gap == analytic omitted", [ref_ml - our_ml], [gap_analytic])
println("\n  => our ML differs from the true ML by ", round(ref_ml - our_ml, digits=3),
        " (a constant for fixed structure; tau-tuning OK, cross-model comparison invalid)")
