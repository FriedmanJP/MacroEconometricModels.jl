using MacroEconometricModels, Distributions, Random
Random.seed!(7)
spec = @dsge begin
    parameters: ρ = 0.9, α = 0.33
    endogenous: Y, K, A
    exogenous: ε
    A[t] = ρ * A[t-1] + ε[t]
    Y[t] = A[t] + α * K[t-1]
    K[t] = 0.9 * K[t-1] + 0.1 * Y[t]
end
spec2 = compute_steady_state(spec)
sol = solve(spec2; method=:gensys)
Y = simulate(sol, 150)
b = estimate_dsge_bayes(spec, Y[:, [1]], [0.8]; priors=Dict(:ρ => Beta(5,2)),
                        method=:mh, n_draws=300, burnin=100, observables=[:Y])
r = irf(b, 10; n_draws=5)
println("irf OK: ", size(r.point_estimate))
f = fevd(b, 10; n_draws=5)
println("fevd OK")
s = simulate(b, 20; n_draws=5)
println("simulate OK")
