using MacroEconometricModels, Random, Statistics, Distributions
sup = MacroEconometricModels._suppress_warnings
sup() do
    rng = Random.MersenneTwister(42)
    spec = @dsge begin
        parameters: ρ = 0.5, σ = 0.05
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    spec = compute_steady_state(spec)
    true_spec = @dsge begin
        parameters: ρ = 0.8, σ = 0.05
        endogenous: y
        exogenous: ε
        y[t] = ρ * y[t-1] + σ * ε[t]
        steady_state = [0.0]
    end
    true_spec = compute_steady_state(true_spec)
    data = simulate(solve(true_spec; method=:gensys), 300; rng=rng)'
    priors = Dict(:ρ => Beta(2, 2), :σ => InverseGamma(3.0, 0.2))
    fit_t = estimate_dsge_bayes(spec, data, [0.5, 0.1]; priors=priors, method=:mh, transform=true,
        n_draws=3000, burnin=1000, observables=[:y], rng=Random.MersenneTwister(7))
    fit_u = estimate_dsge_bayes(spec, data, [0.5, 0.1]; priors=priors, method=:mh, transform=false,
        n_draws=3000, burnin=1000, observables=[:y], rng=Random.MersenneTwister(7))
    mt = vec(mean(fit_t.theta_draws; dims=1)); mu = vec(mean(fit_u.theta_draws; dims=1))
    println("mt = ", mt, "  acc_t=", fit_t.acceptance_rate)
    println("mu = ", mu, "  acc_u=", fit_u.acceptance_rate)
    println("abs(mt[1]-mu[1]) = ", abs(mt[1]-mu[1]))
    println("abs(mt[2]-mu[2]) = ", abs(mt[2]-mu[2]))
    println("std ρ_t=", std(fit_t.theta_draws[:,1]), " ρ_u=", std(fit_u.theta_draws[:,1]))
end
