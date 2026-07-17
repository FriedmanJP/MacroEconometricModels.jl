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
    for (nd, bi) in [(5000,2000), (6000,3000), (8000,4000)]
        fit_t = estimate_dsge_bayes(spec, data, [0.5, 0.1]; priors=priors, method=:mh, transform=true,
            n_draws=nd, burnin=bi, observables=[:y], rng=Random.MersenneTwister(7))
        fit_u = estimate_dsge_bayes(spec, data, [0.5, 0.1]; priors=priors, method=:mh, transform=false,
            n_draws=nd, burnin=bi, observables=[:y], rng=Random.MersenneTwister(7))
        mt = vec(mean(fit_t.theta_draws; dims=1)); mu = vec(mean(fit_u.theta_draws; dims=1))
        println("nd=$nd bi=$bi | mt=", round.(mt;digits=4), " mu=", round.(mu;digits=4),
                " |Δρ|=", round(abs(mt[1]-mu[1]);digits=4), " |Δσ|=", round(abs(mt[2]-mu[2]);digits=4),
                " acc_u=", round(fit_u.acceptance_rate;digits=3))
    end
end
