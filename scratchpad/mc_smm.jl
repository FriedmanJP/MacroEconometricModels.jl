using MacroEconometricModels, Random, Statistics, LinearAlgebra
_sw = MacroEconometricModels._suppress_warnings
_sw() do
    true_rho=0.7; true_sigma=0.5; T_obs=400
    cfn = d -> autocovariance_moment_contributions(d; lags=1)
    function sim_ar1(theta,T_periods,burn;rng=Random.default_rng())
        rho,sigma=theta; sim=zeros(T_periods+burn)
        for t in 2:(T_periods+burn); sim[t]=rho*sim[t-1]+abs(sigma)*randn(rng); end
        reshape(sim[(burn+1):end],:,1)
    end
    bounds=ParameterTransform([-1.0,0.0],[1.0,Inf]); R=160
    rho_hat=Float64[]; se=Float64[]
    for r in 1:R
        drng=Random.MersenneTwister(5000+r); y=zeros(T_obs)
        for t in 2:T_obs; y[t]=true_rho*y[t-1]+true_sigma*randn(drng); end
        res=estimate_smm(sim_ar1, d->autocovariance_moments(d;lags=1),[0.5,0.4],reshape(y,:,1);
            sim_ratio=5,burn=100,weighting=:two_step,contributions_fn=cfn,bounds=bounds,
            rng=Random.MersenneTwister(9000+r))
        push!(rho_hat,res.theta[1]); push!(se,stderror(res)[1])
    end
    println("mc_std(rho)=", std(rho_hat), "  mean_se=", mean(se), "  ratio=", std(rho_hat)/mean(se))
    println("mean(rho_hat)=", mean(rho_hat), "  median_se=", median(se))
    println("n_converged frac approx (rho in (-.99,.99)): ", mean(abs.(rho_hat).<0.99))
end
